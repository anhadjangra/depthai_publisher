#!/usr/bin/env python3
"""
Runs your DepthAI YOLO model and publishes first-known target locations
(translation-only; no yaw) as PointStamped on:
  /marker_locations/id/person
  /marker_locations/id/bag
"""

from pathlib import Path
import time
import json
import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge, CvBridgeError

# =========================== Model / Paths ===========================
pipeline = None
cam_source = 'rgb'  # 'rgb', 'left', 'right'
syncNN = True

modelsPath = "/home/uavteam6/catkin_ws/src/depthai_publisher/src/depthai_publisher/models"
modelName = 'best_openvino_2022.1_6shave'
confJson = 'best.json'

configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
	raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
	config = json.load(f)
nnConfig = config.get("nn_config", {})
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# =========================== Helper mapping ===========================
def canonical_target(label_str):
	"""Map raw YOLO label to the canonical IDs we publish."""
	s = (label_str or "").strip().lower()
	if s == "person":
		return "person"
	if s in ("bag", "backpack", "handbag", "suitcase"):
		return "bag"
	return None

# =========================== Main class ===========================
class DepthaiCamera():
	fps = 10.0

	pub_topic = '/depthai_node/image/compressed'
	pub_topic_raw = '/depthai_node/image/raw'
	pub_topic_detect = '/depthai_node/detection/compressed'
	pub_topic_cam_inf = '/depthai_node/camera/camera_info'

	def __init__(self):
		self.frame_count = 0
		# ---- Params (projection; same as your ArUco node) ----
		self.hfov_deg = float(rospy.get_param('~hfov_deg', 54.0))
		self.vfov_deg = float(rospy.get_param('~vfov_deg', 54.0))
		self.fixed_alt_m = float(rospy.get_param('~fixed_altitude_m', 2.5))
		self.use_live_alt = bool(rospy.get_param('~use_live_altitude', True))
		self.publish_first_only = bool(rospy.get_param('~publish_first_only', False))

		# DepthAI input size (used as our image size for projection)
		self.nn_shape_w, self.nn_shape_h = (416, 416)
		if "input_size" in nnConfig:
			self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

		# Pinhole-ish focal lengths in pixels (derived from FOV and image size)
		self.fpx_h = (self.nn_shape_w / 2.0) / np.tan(np.deg2rad(self.hfov_deg / 2.0))
		self.fpx_v = (self.nn_shape_h / 2.0) / np.tan(np.deg2rad(self.vfov_deg / 2.0))

		# Latest MAVROS local pose (translation-only)
		self.pos_x = None
		self.pos_y = None
		self.altitude_m = None
		self.alt_ema = None
		self.alt_alpha = float(rospy.get_param('~alt_ema_alpha', 0.25))  # small smoothing

		# Track which canonical IDs we've published already (first detection wins)
		self.published_ids = set()  # e.g., {"person", "bag"}

		# ROS pubs
		self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
		self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
		self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
		self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=10)

		# lazy per-ID PointStamped publishers (latched)
		self.id_pubs = {}  # key: "person"/"bag" -> rospy.Publisher

		# CameraInfo timer
		self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

		# Pose subscriber
		rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=10)

		self.br = CvBridge()
		rospy.on_shutdown(lambda: self.shutdown())

		rospy.loginfo("DepthAI YOLO → /marker_locations/id/<person|bag> (first detection only; translation-only).")

		# Build pipeline once
		self.pipeline = self.createPipeline(str((Path(__file__).parent / f"{modelsPath}/{modelName}/{modelName}.blob").resolve()))

	# ---------------- MAVROS pose ----------------
	def pose_cb(self, msg: PoseStamped):
		self.pos_x = msg.pose.position.x
		self.pos_y = msg.pose.position.y
		z = float(msg.pose.position.z)- 0.16
		self.alt_ema = z if self.alt_ema is None else (self.alt_alpha * z + (1 - self.alt_alpha) * self.alt_ema)
		self.altitude_m = self.alt_ema

	# ---------------- CameraInfo ----------------
	def publish_camera_info(self, _evt=None):
		m = CameraInfo()
		m.header.frame_id = "camera_frame"
		m.header.stamp = rospy.Time.now()
		m.height = self.nn_shape_h
		m.width = self.nn_shape_w
		m.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
		m.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
		m.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		m.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
		m.distortion_model = "plumb_bob"
		self.pub_cam_inf.publish(m)

	# ---------------- DepthAI pipeline ----------------
	def createPipeline(self, nnPath):
		p = dai.Pipeline()
		p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

		det = p.create(dai.node.YoloDetectionNetwork)
		det.setConfidenceThreshold(confidenceThreshold)
		det.setNumClasses(classes)
		det.setCoordinateSize(coordinates)
		det.setAnchors(anchors)
		det.setAnchorMasks(anchorMasks)
		det.setIouThreshold(iouThreshold)
		det.setBlobPath(nnPath)
		det.setNumPoolFrames(4)
		det.input.setBlocking(False)
		det.setNumInferenceThreads(2)

		if cam_source == 'rgb':
			cam = p.create(dai.node.ColorCamera)
			cam.setPreviewSize(self.nn_shape_w, self.nn_shape_h)
			cam.setInterleaved(False)
			cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
			cam.setFps(40)
			cam.preview.link(det.input)
		else:
			cam = p.create(dai.node.MonoCamera)
			if cam_source == 'left':
				cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
			else:
				cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
			manip = p.create(dai.node.ImageManip)
			manip.setResize(self.nn_shape_w, self.nn_shape_h)
			manip.setKeepAspectRatio(True)
			manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
			cam.out.link(manip.inputImage)
			manip.out.link(det.input)

		xout_rgb = p.create(dai.node.XLinkOut)
		xout_rgb.setStreamName("nn_input")
		xout_rgb.input.setBlocking(False)
		det.passthrough.link(xout_rgb.input)

		xout_det = p.create(dai.node.XLinkOut)
		xout_det.setStreamName("nn")
		xout_det.input.setBlocking(False)
		det.out.link(xout_det.input)

		return p

	# ---------------- Publishing helpers ----------------
	def publish_to_ros(self, frame):
		msg_out = CompressedImage()
		msg_out.header.stamp = rospy.Time.now()
		msg_out.format = "jpeg"
		msg_out.header.frame_id = "camera_frame"
		msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
		self.pub_image.publish(msg_out)
		self.pub_image_raw.publish(self.br.cv2_to_imgmsg(frame, encoding="bgr8"))

	def publish_detect_to_ros(self, frame):
		msg_out = CompressedImage()
		msg_out.header.stamp = rospy.Time.now()
		msg_out.format = "jpeg"
		msg_out.header.frame_id = "camera_frame"
		msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
		self.pub_image_detect.publish(msg_out)

	def publish_point_once(self, target_id, x_map, y_map):
		"""Publish a latched PointStamped once per target_id ('person' or 'bag')."""
		if target_id in self.published_ids and self.publish_first_only:
			return
		if target_id not in self.id_pubs:
			topic = f"/marker_locations/id/{target_id}"
			self.id_pubs[target_id] = rospy.Publisher(topic, PointStamped, queue_size=1, latch=True)
			rospy.loginfo(f"[YOLO] Created per-ID topic: {topic}")

		msg = PointStamped()
		msg.header.stamp = rospy.Time.now()
		msg.header.frame_id = "map"  # local world frame
		msg.point.x = float(x_map)
		msg.point.y = float(y_map)
		msg.point.z = 0.0
		self.id_pubs[target_id].publish(msg)
		self.published_ids.add(target_id)

	# ---------------- Projection math (no yaw) ----------------
	def bbox_center_uv(self, det):
		"""Return bbox center (u, v) in pixel coords of the NN frame."""
		# DepthAI gives [0..1]; scale by frame size
		u = ((det.xmin + det.xmax) * 0.5) * self.nn_shape_w
		v = ((det.ymin + det.ymax) * 0.5) * self.nn_shape_h
		return float(u), float(v)

	def project_to_map_translation_only(self, u, v):
		"""Project (u,v) to (x_map,y_map) with ground-plane assumption and translation-only."""
		# choose height
		if self.use_live_alt and (self.altitude_m is not None) and (self.altitude_m > 0.0):
			h = float(self.altitude_m)
		else:
			h = float(self.fixed_alt_m)

		cx = self.nn_shape_w / 2.0
		cy = self.nn_shape_h / 2.0

		# Your convention (image-centered): top=+x_cam, left=+y_cam
		x_px = (cy - v)   # up is +x
		y_px = (cx - u)   # left is +y

		# meters in camera ground-plane
		x_cam = h * (x_px / self.fpx_v)
		y_cam = h * (y_px / self.fpx_h)

		# translation-only into local ENU (map)
		px = float(self.pos_x) if self.pos_x is not None else 0.0
		py = float(self.pos_y) if self.pos_y is not None else 0.0
		x_map = px + x_cam - 0.1
		y_map = py + y_cam
		return x_map, y_map

	# ---------------- Main loop ----------------
	def run(self):
		# Open the device and start the (already-built) pipeline ONCE
		with dai.Device(self.pipeline) as device:
			cams = device.getConnectedCameras()
			depth_enabled = (dai.CameraBoardSocket.LEFT in cams) and (dai.CameraBoardSocket.RIGHT in cams)
			if cam_source != "rgb" and not depth_enabled:
				raise RuntimeError("Unable to run on {} camera! Available: {}".format(cam_source, cams))

			

			# Output queues from the pipeline
			q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
			q_nn       = device.getOutputQueue(name="nn",       maxSize=4, blocking=False)

			# FPS accounting
			start_time = time.time()
			counter = 0
			fps = 0.0
			frame_counter = 0
			# Main processing loop
			while not rospy.is_shutdown():
				inRgb = q_nn_input.tryGet()
				inDet = q_nn.tryGet()

				if inRgb is None or inDet is None:
					# nothing ready this tick; sleep a touch to avoid busy spin
					time.sleep(0.001)
					continue
				frame_counter +=1
				if frame_counter %2 !=0:
					continue

				# Frames & detections
				frame = inRgb.getCvFrame()
				detections = inDet.detections

				# Draw/annotate & publish detections
				overlay = frame.copy()
				found_classes = set()

				for det in detections:
					# Map raw label -> canonical ('person' / 'bag'), skip others
					lbl = labels[det.label] if det.label < len(labels) else str(det.label)
					canon = canonical_target(lbl)
					if canon is None:
						continue
					found_classes.add(canon)

					# Draw bbox & label
					x1, y1, x2, y2 = self.frameNorm(overlay, (det.xmin, det.ymin, det.xmax, det.ymax))
					cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
					cv2.putText(overlay, f"{canon} {int(det.confidence*100)}%",
								(x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

					# Project bbox center → map (translation-only; no yaw)
					u, v = self.bbox_center_uv(det)
					x_map, y_map = self.project_to_map_translation_only(u, v)

					# Annotate projected coords
					cv2.putText(overlay, f"map({x_map:.2f},{y_map:.2f})",
								(x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

					# Publish once per canonical ID (latched)
					self.publish_point_once(canon, x_map, y_map)

				# Update FPS
				counter += 1
				now = time.time()
				elapsed = now - start_time
				if elapsed >= 1.0:
					fps = counter / elapsed
					counter = 0
					start_time = now

				# HUD and image publishing
				cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 6),
							cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
				if found_classes:
					cv2.putText(overlay, "Found: {}".format(sorted(found_classes)), (2, 14),
								cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))

				self.publish_to_ros(frame)
				self.publish_detect_to_ros(overlay)

	# Utilities
	def frameNorm(self, frame, bbox):
		normVals = np.full(len(bbox), frame.shape[0])
		normVals[::2] = frame.shape[1]
		return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

	def shutdown(self):
		cv2.destroyAllWindows()


# ------------------------- Main -------------------------
def main():
	rospy.init_node('depthai_node')
	dai_cam = DepthaiCamera()
	try:
		dai_cam.run()
	finally:
		dai_cam.shutdown()


if __name__ == "__main__":
	main()

