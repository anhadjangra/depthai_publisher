#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ArucoDetector():
    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        # --- Choose your dictionary here ---
        dict_id = cv2.aruco.DICT_5X5_100   # change to DICT_5X5_250 if you use 5x5 markers

        # --- Handle both new and old OpenCV ArUco APIs ---
        try:
            # New API (OpenCV >= 4.7)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_api = True
        except AttributeError:
            # Old API fallback (OpenCV <= 4.6)
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False

        self.aruco_pub = rospy.Publisher('/processed_aruco/image/compressed',
                                         CompressedImage, queue_size=10)
        self.br = CvBridge()

        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(self.frame_sub_topic,
                                              CompressedImage,
                                              self.img_callback,
                                              queue_size=1, buff_size=2**24)

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        annotated = self.find_aruco(frame)
        self.publish_to_ros(annotated)

    def find_aruco(self, frame):
        # Detect on grayscale for robustness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                      parameters=self.aruco_params)

        if ids is not None and len(corners) > 0:
            ids = ids.flatten()

            # Quick overlay (also draws IDs if provided)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Optional: custom box + ID text (kept close to your original)
            for (marker_corner, marker_ID) in zip(corners, ids):
                pts = marker_corner.reshape((4, 2)).astype(int)
                (tl, tr, br, bl) = [tuple(p) for p in pts]

                cv2.line(frame, tl, tr, (0, 255, 0), 2)
                cv2.line(frame, tr, br, (0, 255, 0), 2)
                cv2.line(frame, br, bl, (0, 255, 0), 2)
                cv2.line(frame, bl, tl, (0, 255, 0), 2)

                rospy.loginfo_throttle(1.0, f"Aruco detected, ID: {marker_ID}")
                cv2.putText(frame, str(marker_ID),
                            (tl[0], max(0, tl[1] - 10)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def publish_to_ros(self, frame):
        ok, enc = cv2.imencode('.jpg', frame)
        if not ok:
            rospy.logwarn("Failed to encode JPEG")
            return
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = enc.tobytes()   # tostring() is deprecated
        self.aruco_pub.publish(msg_out)
        
def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")
    _ = ArucoDetector()
    rospy.spin()

if __name__ == "__main__":
    main()
