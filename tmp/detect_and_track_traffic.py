# detect_and_track_traffic.py
# December 2020
# Read a video, make detections on frames, and track detected objects across frames.
# Draw the detections on the frames, prepare the output, and save the processed video.


from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from videostream import Video_Thread
from mobilenet import Mnet
import numpy as np
import cviz
import dlib
import time
import cv2
import sys
import os


class Traffic_Detection:
	def __init__(self, width=None, freq=5):
		self.detect_freq = freq
		self.resize_width = width
		self.mn = Mnet()
		self.net = self.mn.net
		self.frame_count = 0
		self.trackers = []
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 32)
		self.w = 0
		self.h = 0
		self.statuses = {}


	def detect(self, frame, rgb, v_only=True):
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
			if class_idx not in self.mn.traffic_idxs:
				continue
			if v_only and class_idx not in self.mn.vehicles_only:
				continue
			if class_idx not in self.mn.vehicles_only and confidence < 0.6:
				continue
			label = self.mn.all_classes[class_idx]
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


	def traffic_detections(self, frame):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		boxs = []
		if self.frame_count % self.detect_freq == 0:
			self.detect(frame, rgb)
		else:
			for tracker, label in self.trackers:
				boxs = self.track(tracker, label, boxs, rgb)
		objects = self.ct.update(boxs)
		for (objectID, (c, box, label)) in objects.items():
			to = self.trackable_objects.get(objectID, None)
			if to is None:
				to = Trackable_Object(objectID, c)
				to.label = label
				to.color = self.mn.colors[to.label]
			else:
				to.centroids.append(c)
			self.trackable_objects[objectID] = to
			(start_x, start_y, end_x, end_y) = box
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), to.color, 2)
			overlay = frame.copy()
			radius = min( (end_x - start_x) // 2, (end_y - start_y) // 2 )
			cv2.circle(overlay, (c[0], c[1]), (radius), to.color, -1)
			frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, 0)
			cv2.circle(frame, (c[0], c[1]), (radius), to.color, 1)
			cv2.putText(frame, str(objectID), (start_x+10, start_y+15), 0, 0.55, (255,255,255), 2)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
		cv2.putText(frame, f"{self.frame_count}", (10, 10), 0, 0.35, (20,255,10), 1)
		return frame


	def read_video(self, vid_path, output):
		self.w, self.h = cviz.vid_dimz(vid_path, self.resize_width)
		vs = Video_Thread(vid_path).start()
		self.set_status(vs.frames())
		writer = cviz.vid_writer(output, self.w, self.h, vs.fps())
		while True:
			frame = vs.read()
			if frame is None:
				break
			if self.resize_width:
				frame = cviz.resize(frame, width=self.resize_width)
			frame = self.traffic_detections(frame)
			writer.write(frame)
			self.status()
			self.frame_count += 1
		vs.release()


	def set_status(self, frames):
		print(f"Processing {frames} video frames")
		self.statuses = {
			int(frames*0.25) : "25% complete",
			int(frames*0.50) : "50% complete",
			int(frames*0.75) : "75% complete",
		}


	def status(self):
		if self.frame_count in self.statuses:
			print(self.statuses.get(self.frame_count))



##
