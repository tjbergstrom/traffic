# detect_and_track_traffic.py
# December 2020


import dlib
import numpy as np
from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from videostream import Video_Thread
from mobilenet import Mnet
import imutils
import time
import cv2
import sys
import os


class Traffic_Detection:
	def __init__(self):
		self.trackers = []
		self.detect_freq = 3
		self.frame_count = 0
		self.mobile_net = Mnet()
		self.net = self.mobile_net.net
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 32)
		self.w = None
		self.h = None


	def detect(self, frame, rgb):
		self.trackers = []
		blob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		self.net.setInput(blob)
		detections = self.net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence < 0.4:
				continue
			class_idx = int(detections[0, 0, i, 1])
			if class_idx not in self.mobile_net.traffic_idxs:
				continue
			label = self.mobile_net.all_classes[class_idx]
			box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			tracker = dlib.correlation_tracker()
			box = dlib.rectangle(start_x, start_y, end_x, end_y)
			tracker.start_track(rgb, box)
			self.trackers.append((tracker, label))





	def track(self, tracker, label, boxs, rgb):
		tracker.update(rgb)
		pos = tracker.get_position()
		start_x = int(pos.left())
		start_y = int(pos.top())
		end_x = int(pos.right())
		end_y = int(pos.bottom())
		boxs.append((start_x, start_y, end_x, end_y, label))
		return boxs




	def traffic_detections(self, frame, rgb):
		boxs = []
		if self.frame_count % self.detect_freq == 0:
			self.detect(frame, rgb)
		else:
			for tracker, label in self.trackers:
				boxs = self.track(tracker, label, boxs, rgb)
		objects = self.ct.update(boxs)
		for (objectID, (centroid, box, label)) in objects.items():
			to = self.trackable_objects.get(objectID, None)
			if to is None:
				to = Trackable_Object(objectID, centroid)
				to.label = label
				try:
					to.color = self.mobile_net.clrs[to.label]
					print(objectID, (centroid, box, label))
				except:
					print(objectID, (centroid, box, label))
					sys.exit(1)
			else:
				to.centroids.append(centroid)
			self.trackable_objects[objectID] = to
			(start_x, start_y, end_x, end_y) = box
			cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), to.color, 2)
			cv2.putText(frame, str(objectID), (centroid[0], centroid[1]), 0, 2, to.color, 3)
		return frame




	def read_video(self, vid_path):
		vs = Video_Thread(vid_path).start()
		while True:
			frame = vs.read()
			if frame is None:
				break
			frame = imutils.resize(frame, width=720)
			if self.w is None or self.h is None:
				(self.h, self.w) = frame.shape[:2]
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = self.traffic_detections(frame, rgb)
			self.frame_count += 1
			cv2.imshow("Video", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				vs.release()
				vs.stop()
				break



if __name__ == '__main__':
	TD = Traffic_Detection()
	TD.read_video("vid_inputs/vid8.mp4")
