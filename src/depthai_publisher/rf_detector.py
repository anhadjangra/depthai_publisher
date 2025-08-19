#!/usr/bin/env python3
import os
import io
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from inference import get_model

class RFDetectorNode:
    def __init__(self):
        # Params (can be overridden with ROS params)
        self.model_id = rospy.get_param("~model_id", "sam-tixx2/image-detector-wuwng-instant/3")
        self.api_key  = rospy.get_param("~api_key", "2ypgrFnH0imAupDBxz5e")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image/compressed")
        self.score_thresh = float(rospy.get_param("~score_thresh", 0.35))
        self.classes = rospy.get_param("~classes", ["bag", "person"])  # used to map names -> IDs

        if not self.api_key:
            rospy.logfatal("No API key set. Pass ~api_key or set ROBOFLOW_API_KEY.")
            raise SystemExit(1)

        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        rospy.loginfo(f"Loading Roboflow model: {self.model_id}")
        self.model = get_model(self.model_id, api_key=self.api_key)

        self.bridge = CvBridge()
        self.pub = rospy.Publisher("detections", Detection2DArray, queue_size=1)

        # Subscribe to either raw or compressed image
        if self.image_topic.endswith("/compressed"):
            self.sub = rospy.Subscriber(self.image_topic, CompressedImage, self.cb_compressed, queue_size=1, buff_size=2**22)
        else:
            self.sub = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**22)

    def cb_compressed(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        import cv2
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.run_infer_and_publish(frame, msg.header)

    def cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.run_infer_and_publish(frame, msg.header)

    def run_infer_and_publish(self, frame, header):
        preds = self.model.infer(frame)[0].get("predictions", [])
        arr = Detection2DArray()
        arr.header = header

        for p in preds:
            conf = float(p.get("confidence", 0.0))
            if conf < self.score_thresh:
                continue

            # Roboflow returns center-x/center-y/width/height in pixels
            cx, cy = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            cls_name = p.get("class", "unknown")
            cls_id = self.class_to_id.get(cls_name, -1)

            det = Detection2D()
            det.header = header
            det.bbox.center.x = cx
            det.bbox.center.y = cy
            det.bbox.size_x = w
            det.bbox.size_y = h

            hyp = ObjectHypothesisWithPose()
            # vision_msgs for ROS Noetic expects a numeric class id; map your class names to ints
            hyp.id = int(cls_id)
            hyp.score = conf
            det.results.append(hyp)
            arr.detections.append(det)

        self.pub.publish(arr)

if __name__ == "__main__":
    rospy.init_node("rf_detector")
    RFDetectorNode()
    rospy.loginfo("rf_detector node started.")
    rospy.spin()
