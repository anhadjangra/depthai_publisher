#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
import threading
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict, deque

class ArucoDetector():
    # Topics (can be overridden by ROS params)
    frame_sub_topic_compressed = '/depthai_node/image/compressed'
    frame_sub_topic_raw        = '/depthai_node/image/raw'
    pose_sub_topic             = '/mavros/local_position/pose'
    out_image_topic            = '/processed_aruco/image/compressed'
    out_markers_topic          = '/marker_locations'

    def __init__(self):
        self.frame_count = 0
        # ---- Params ----
        self.hfov_deg = float(rospy.get_param('~hfov_deg', 54.0))
        self.vfov_deg = float(rospy.get_param('~vfov_deg', 54.0))
        self.fixed_alt_m = float(rospy.get_param('~fixed_altitude_m', 2.5))
        self.use_live_alt = bool(rospy.get_param('~use_live_altitude', True))
        self.avg_window = int(rospy.get_param('~avg_window', 10))
        self.publish_first_only = bool(rospy.get_param('~publish_first_only', False))
        self.use_raw_image = bool(rospy.get_param('~use_raw_image', False))
        self.alt_ema_alpha = float(rospy.get_param('~alt_ema_alpha', 0.25))  # altitude smoothing

        # Image intrinsics (computed on first frame)
        self.W = None; self.H = None
        self.fpx_h = None; self.fpx_v = None

        # Latest pose (translation-only). Guarded by a lock since callbacks are async.
        self._pose_lock = threading.Lock()
        self.pos_x = None
        self.pos_y = None
        self.altitude_m = None
        self._alt_ema = None

        # For optional smoothing/averaging later if you want
        self.hist = defaultdict(lambda: deque(maxlen=self.avg_window))

        # One-shot per ID publishing
        self.id_pubs = {}          # id -> rospy.Publisher(PointStamped)
        self.published_ids = set() # IDs already published if publish_first_only=True

        # --- ArUco setup ---
        dict_id = cv2.aruco.DICT_5X5_100
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_api = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False

        # Publishers
        self.aruco_pub = rospy.Publisher(self.out_image_topic, CompressedImage, queue_size=10)
        self.marker_pub = rospy.Publisher(self.out_markers_topic, Float32MultiArray, queue_size=10)

        self.br = CvBridge()

        # Subscribers (NO SYNC)
        if not rospy.is_shutdown():
            if self.use_raw_image:
                rospy.loginfo("[ArucoDetector] Subscribing to RAW images: %s", self.frame_sub_topic_raw)
                rospy.Subscriber(self.frame_sub_topic_raw, Image, self.img_cb,
                                 queue_size=1, buff_size=2**24)
            else:
                rospy.loginfo("[ArucoDetector] Subscribing to COMPRESSED images: %s", self.frame_sub_topic_compressed)
                rospy.Subscriber(self.frame_sub_topic_compressed, CompressedImage, self.img_cb,
                                 queue_size=1, buff_size=2**24)

            rospy.Subscriber(self.pose_sub_topic, PoseStamped, self.pose_cb, queue_size=10)

        rospy.loginfo("[ArucoDetector] Sync-free; using latest pose. Translation-only; first-hit per ID.")

    # -------- Pose callback (stores latest) --------
    def pose_cb(self, msg: PoseStamped):
        with self._pose_lock:
            self.pos_x = float(msg.pose.position.x)
            self.pos_y = float(msg.pose.position.y)
            z = float(msg.pose.position.z)-0.16
            self._alt_ema = z if self._alt_ema is None else (self.alt_ema_alpha * z + (1 - self.alt_ema_alpha) * self._alt_ema)
            self.altitude_m = self._alt_ema

    # -------- Image callback (process every image) --------
    def img_cb(self, img_msg):
        self.frame_count +=1
        if self.frame_count %2 != 0:
            return

        # Decode image
        try:
            if isinstance(img_msg, CompressedImage):
                frame = self.br.compressed_imgmsg_to_cv2(img_msg)
            else:
                frame = self.br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("[ArucoDetector] CvBridgeError: %s", str(e))
            return
        except Exception as e:
            rospy.logerr("[ArucoDetector] Unexpected decode error: %s", str(e))
            return

        self._ensure_intrinsics(frame)

        # Copy latest pose atomically
        with self._pose_lock:
            px = self.pos_x
            py = self.pos_y
            h  = self.altitude_m

        # Fallbacks if pose not yet available
        if not self.use_live_alt or h is None or h <= 0.0:
            h = float(self.fixed_alt_m)
        if px is None: px = 0.0
        if py is None: py = 0.0

        annotated_frame, new_ids_array = self.find_aruco_and_project_first_only(frame, px, py, h)

        # Put a banner if we are using fallback pose/altitude
        if (self.pos_x is None) or (self.pos_y is None) or (self.altitude_m is None):
            cv2.putText(annotated_frame, "NO LIVE POSE: using (0,0) and/or fixed altitude",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

        # Publish annotated image every frame
        self.publish_image(annotated_frame)

        # Optionally publish aggregate array (new IDs only this frame)
        if new_ids_array:
            self.marker_pub.publish(Float32MultiArray(data=new_ids_array))

    def _ensure_intrinsics(self, frame):
        if self.W is None or self.H is None:
            self.H, self.W = frame.shape[:2]
            self.fpx_h = (self.W / 2.0) / np.tan(np.deg2rad(self.hfov_deg / 2.0))
            self.fpx_v = (self.H / 2.0) / np.tan(np.deg2rad(self.vfov_deg / 2.0))
            rospy.loginfo(f"[ArucoDetector] Image {self.W}x{self.H}, fpx_h={self.fpx_h:.2f}, fpx_v={self.fpx_v:.2f}")

    def find_aruco_and_project_first_only(self, frame, px, py, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        triples_new = []  # [x_map, y_map, id, ...] ONLY for newly published IDs

        if ids is not None and len(corners) > 0:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            cx = self.W / 2.0
            cy = self.H / 2.0

            for (marker_corner, marker_id) in zip(corners, ids):
                # Only publish the FIRST detection per ID if enabled
                if self.publish_first_only and (marker_id in self.published_ids):
                    continue

                pts = marker_corner.reshape((4, 2)).astype(float)
                (tl, tr, br, bl) = pts

                u = np.mean([tl[0], tr[0], br[0], bl[0]])  # column (right +)
                v = np.mean([tl[1], tr[1], br[1], bl[1]])  # row (down +)

                # Image-centered pixel offsets: top=+x_cam, left=+y_cam
                x_px = (cy - v)
                y_px = (cx - u)

                # Project to meters on ground at height h
                x_cam = h * (x_px / self.fpx_v)
                y_cam = h * (y_px / self.fpx_h)

                # Translation-only into local map/ENU (no rotation)
                x_map = px + x_cam -0.1
                y_map = py + y_cam

                # Publish per-ID (latched)
                if self.publish_first_only:
                    self.published_ids.add(marker_id)
                self.publish_per_id(mid=marker_id, x_map=x_map, y_map=y_map)

                # annotate
                cpt = (int(round(u)), int(round(v)))
                cv2.circle(frame, cpt, 4, (0, 255, 255), -1)
                tag = "[FIRST]" if self.publish_first_only else ""
                cv2.putText(frame, f"map({x_map:.2f},{y_map:.2f}) id={int(marker_id)} {tag}",
                            (cpt[0] + 5, cpt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                # Aggregate array (new IDs only this frame)
                triples_new.extend([float(x_map), float(y_map), float(marker_id)])

        return frame, triples_new

    def publish_per_id(self, mid, x_map, y_map):
        """
        Publish a latched PointStamped (once per ID if publish_first_only=True) on:
          /marker_locations/id/<id>
        """
        if mid not in self.id_pubs:
            topic = f"/marker_locations/id/{mid}"
            self.id_pubs[mid] = rospy.Publisher(topic, PointStamped, queue_size=1, latch=True)
            rospy.loginfo(f"[ArucoDetector] Created per-ID topic: {topic}")

        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.point.x = float(x_map)
        msg.point.y = float(y_map)
        msg.point.z = 0.0
        self.id_pubs[mid].publish(msg)
    def publish_image(self, frame):
        ok, enc = cv2.imencode('.jpg', frame)
        if not ok:
            rospy.logwarn("[ArucoDetector] Failed to encode JPEG")
            return
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.header.frame_id = "camera_frame"
        msg_out.format = "jpeg"
        msg_out.data = enc.tobytes()
        self.aruco_pub.publish(msg_out)

def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Aruco: sync-free; using latest pose; translation-only; first-hit per ID.")
    _ = ArucoDetector()
    rospy.spin()

if __name__ == "__main__":
    main()
