# detect_and_track_traffic.py
# December 2020
# Read a video, make detections on frames, and track detected objects across frames.
# Draw the detections on the frames and prepare the output for displaying and saving.


from traffyc.centroid_tracker import Centroid_Tracker
from traffyc.trackable_object import Trackable_Object
from traffyc.videostream import Video_Thread
from traffyc.mobilenet import Mnet
import numpy as np
import imutils
import dlib
import time
import cv2
import sys
import os
import random


class Traffic_Detection:
	def __init__(self, width=None, freq=5):
		self.trackers = []
		self.detect_freq = freq
		self.frame_count = 0
		self.mobile_net = Mnet()
		self.net = self.mobile_net.net
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 32)
		self.resize_width = width
		self.w = 0
		self.h = 0
		#f = open("mpt/points.csv", "r")
		#self.point_frames = f.readlines()
		#self.point_frames = []
		#f.close()


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
			if class_idx not in self.mobile_net.traffic_idxs:
				continue
			if v_only and class_idx not in self.mobile_net.vehicles_only:
				continue
			if class_idx not in self.mobile_net.vehicles_only and confidence < 0.6:
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
		#self.point_frames.append([])
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
				to.color = self.mobile_net.colors[to.label]
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
			#cv2.putText(frame, str(objectID), (c[0], c[1]), 0, 1.5, to.color, 3)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
			##frame = self.save_points(frame, c, box)
		return frame


	def save_points(self, c, box):
		from scipy.spatial import distance as dist
		(start_x, start_y, end_x, end_y) = box
		radius = min( (end_x - start_x) // 2, (end_y - start_y) // 2 )
		for i in range(128):
			x = int(random.uniform(start_x, end_x))
			y = int(random.uniform(start_y, end_y))
			r_2 = dist.euclidean((x, y), (c[0], c[1]))
			if r_2 <= radius:
				self.point_frames[self.frame_count].append(f"{x} {y}")


	def draw_points(self, frame):
		colrs = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
		point_frame = self.point_frames[self.frame_count].replace("\n", "")
		for point in point_frame.split(","):
			if point:
				x = int(point.split(" ")[0])
				y = int(point.split(" ")[1])
				c = random.randrange(0, len(colrs))
				cv2.circle(frame, (x, y), 3, colrs[c], -1)
		return frame




	def read_video(self, vid_path, output, play):
		self.w, self.h, frames = Video_Thread.vid_dims(vid_path, self.resize_width)
		vs = Video_Thread(vid_path, frames).start()
		if output:
			writer = vs.vid_writer(output, self.w, self.h)
		while True:
			frame = vs.read()
			if frame is None:
				break
			if self.resize_width:
				frame = imutils.resize(frame, width=self.resize_width)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = self.traffic_detections(frame, rgb)
			#frame = self.draw_points(frame)
			self.frame_count += 1
			if output:
				writer.write(frame)
				vs.status(self.frame_count)
			if play:
				cv2.imshow("", frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
		vs.release()

		'''
		f = open("mpt/points.csv", "w")
		for point_frame in self.point_frames:
			for points in point_frame:
				f.write(f"{points},")
			f.write("\n")
		f.close()
		'''



##
