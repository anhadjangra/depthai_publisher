#!/usr/bin/env python3


##Intended usage: 
#Python+++++++++++++++++++++++++++++++++++++++
# import rospy
# from geometry_msgs.msg import PointStamped

# def cb(msg):
#     print("x:", msg.point.x, "y:", msg.point.y)

# rospy.init_node("listen_marker_23")
# rospy.Subscriber("/marker_locations/id/23", PointStamped, cb)
# rospy.spin()
#Bash++++++++++++++++++++++++++++++
#rostopic echo /marker_locations/id/23
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict, deque



class ArucoDetector():
    frame_sub_topic = '/depthai_node/image/compressed'
    pose_sub_topic  = '/mavros/vision_pose/pose'

    def __init__(self):
        # ---- Params ----
        self.hfov_deg = rospy.get_param('~hfov_deg', 54.0)
        self.vfov_deg = rospy.get_param('~vfov_deg', 54.0)
        self.fixed_alt_m = rospy.get_param('~fixed_altitude_m', 2.5)
        self.use_live_alt = rospy.get_param('~use_live_altitude', True)
        self.avg_window = int(rospy.get_param('~avg_window', 10))

        # Image intrinsics
        self.W = None
        self.H = None
        self.fpx_h = None
        self.fpx_v = None

        # Pose/altitude
        self.altitude_m = None

        # Sliding buffers: id -> deque of (x, y)
        self.hist = defaultdict(lambda: deque(maxlen=self.avg_window))

        # Per-ID publishers (latched) for easy access
        self.id_pubs = {}  # id -> rospy.Publisher(PointStamped)

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
        self.aruco_pub = rospy.Publisher('/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        self.marker_pub = rospy.Publisher('/marker_locations', Float32MultiArray, queue_size=10)

        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.pose_sub  = rospy.Subscriber(self.pose_sub_topic, PoseStamped, self.pose_callback, queue_size=1)
            self.frame_sub = rospy.Subscriber(self.frame_sub_topic, CompressedImage, self.img_callback,
                                              queue_size=1, buff_size=2**24)

    def pose_callback(self, msg):
        self.altitude_m = msg.pose.position.z

    def _ensure_intrinsics(self, frame):
        if self.W is None or self.H is None:
            self.H, self.W = frame.shape[:2]
            self.fpx_h = (self.W / 2.0) / np.tan(np.deg2rad(self.hfov_deg / 2.0))
            self.fpx_v = (self.H / 2.0) / np.tan(np.deg2rad(self.vfov_deg / 2.0))
            rospy.loginfo(f"[ArucoDetector] Image {self.W}x{self.H}, fpx_h={self.fpx_h:.2f}, fpx_v={self.fpx_v:.2f}")

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self._ensure_intrinsics(frame)

        annotated_frame, raw_triples = self.find_aruco_and_project(frame)
        avg_triples = self.update_and_average(raw_triples)

        self.publish_image(annotated_frame)
        self.publish_markers(avg_triples)       # aggregate
        self.publish_per_id(avg_triples)        # per-ID PointStamped (latched)

    def find_aruco_and_project(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        triples = []  # raw [x,y,id, ...] (meters)

        if ids is not None and len(corners) > 0:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Height to use
            if self.use_live_alt and (self.altitude_m is not None) and (self.altitude_m > 0.0):
                h = float(self.altitude_m)
            else:
                h = float(self.fixed_alt_m)

            cx = self.W / 2.0
            cy = self.H / 2.0

            for (marker_corner, marker_id) in zip(corners, ids):
                pts = marker_corner.reshape((4, 2)).astype(float)
                (tl, tr, br, bl) = pts

                u = np.mean([tl[0], tr[0], br[0], bl[0]])  # column
                v = np.mean([tl[1], tr[1], br[1], bl[1]])  # row

                # Convention: top=+x, left=+y (relative to center)
                x_px = (cy - v)
                y_px = (cx - u)

                x_m = h * (x_px / self.fpx_v)
                y_m = h * (y_px / self.fpx_h)

                triples.extend([float(x_m), float(y_m), float(marker_id)])

                # annotate
                cpt = (int(round(u)), int(round(v)))
                cv2.circle(frame, cpt, 4, (0, 255, 255), -1)
                cv2.putText(frame, f"raw({x_m:.2f},{y_m:.2f}) id={int(marker_id)}",
                            (cpt[0] + 5, cpt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return frame, triples

    def update_and_average(self, raw_triples):
        averaged = []
        for i in range(0, len(raw_triples), 3):
            x = raw_triples[i + 0]
            y = raw_triples[i + 1]
            mid = int(raw_triples[i + 2])

            self.hist[mid].append((x, y))
            arr = np.array(self.hist[mid], dtype=float)
            x_avg, y_avg = arr.mean(axis=0).tolist()
            averaged.extend([x_avg, y_avg, float(mid)])

        return averaged

    def publish_markers(self, triples):
        # Aggregate array: [x_avg, y_avg, id, ...]
        self.marker_pub.publish(Float32MultiArray(data=triples))

    def publish_per_id(self, triples):
        """
        For each (x_avg, y_avg, id) in THIS frame, publish a latched PointStamped on:
          /marker_locations/id/<id>
        So consumers can just subscribe to the ID they care about.
        """
        t = rospy.Time.now()
        for i in range(0, len(triples), 3):
            x = float(triples[i + 0])
            y = float(triples[i + 1])
            mid = int(triples[i + 2])

            # Build (or reuse) publisher for this ID
            if mid not in self.id_pubs:
                topic = f"/marker_locations/id/{mid}"
                self.id_pubs[mid] = rospy.Publisher(topic, PointStamped, queue_size=1, latch=True)
                rospy.loginfo(f"[ArucoDetector] Created per-ID topic: {topic}")

            msg = PointStamped()
            msg.header.stamp = t
            msg.header.frame_id = "map"  # world-aligned per your convention
            msg.point.x = x
            msg.point.y = y
            msg.point.z = 0.0
            self.id_pubs[mid].publish(msg)

    def publish_image(self, frame):
        ok, enc = cv2.imencode('.jpg', frame)
        if not ok:
            rospy.logwarn("Failed to encode JPEG")
            return
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = enc.tobytes()
        self.aruco_pub.publish(msg_out)

def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Publishing averaged marker positions and per-ID topics under /marker_locations/id/<ID> ...")
    _ = ArucoDetector()
    rospy.spin()

if __name__ == "__main__":
    main()
