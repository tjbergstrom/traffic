# detectrack.py
# May 2020
# Take input video frames, make detections on the frames, track detected objects across frames,
# Draw the detections on the frames, and return the processed output frames.


from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from mobilenet import Mnet
import numpy as np
import dlib
import cv2


class Traffic_Detection:
	def __init__(self, w, h, freq=5, jump_unit=None):
		self.w = w
		self.h = h
		self.MN = Mnet()
		self.trackers = []
		self.frame_count = 0
		self.detect_freq = freq
		self.jump_unit = jump_unit
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 64)


	def mobilenet_detect(self, frame, rgb, v_only=True):
		self.trackers = []
		blob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		self.MN.net.setInput(blob)
		detections = self.MN.net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence < 0.4:
				continue
			class_idx = int(detections[0, 0, i, 1])
			if class_idx not in self.MN.traffic_idxs:
				continue
			if v_only and class_idx not in self.MN.vehicles_only:
				continue
			if class_idx not in self.MN.vehicles_only and confidence < 0.6:
				continue
			box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			tracker = dlib.correlation_tracker()
			tracker.start_track(rgb, dlib.rectangle(start_x, start_y, end_x, end_y))
			self.trackers.append((tracker, self.MN.all_classes[class_idx]))


	def track(self, tracker, label, boxs, rgb):
		tracker.update(rgb)
		pos = tracker.get_position()
		start_x, start_y = int(pos.left()), int(pos.top())
		end_x, end_y = int(pos.right()), int(pos.bottom())
		boxs.append((start_x, start_y, end_x, end_y, label))
		return boxs


	def traffic_detections(self, frame, obj_id=False):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		boxs = []
		if self.frame_count % self.detect_freq == 0:
			self.mobilenet_detect(frame, rgb)
		else:
			for tracker, label in self.trackers:
				boxs = self.track(tracker, label, boxs, rgb)
		objects = self.ct.update(boxs)
		for (objectID, (c, box, label)) in objects.items():
			to = self.trackable_objects.get(objectID, None)
			if to is None:
				to = Trackable_Object(objectID, c)
				to.label = label
				to.color = self.MN.colors[to.label]
			else:
				to.centroids.append(c)
			self.trackable_objects[objectID] = to
			(start_x, start_y, end_x, end_y) = box
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), to.color, 1)
			overlay = frame.copy()
			radius = min( (end_x - start_x) // 2, (end_y - start_y) // 2 )
			cv2.circle(overlay, (c[0], c[1]), (radius), to.color, -1)
			frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, 0)
			cv2.circle(frame, (c[0], c[1]), (radius), to.color, 1)
			if obj_id:
				cv2.putText(frame, f"{objectID}", (c[0], c[1]), 0, 1.5, to.color, 3)
			cv2.putText(frame, to.label, (start_x+10,start_y+15), 0, 0.5, to.color, 2)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
		return frame



##
