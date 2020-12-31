# detect_and_track_traffic.py
# December 2020


from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from videostream import Video_Thread
from mobilenet import Mnet
import numpy as np
import imutils
import dlib
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
			if class_idx != 7 and confidence < 8:
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
		for (objectID, (c, box, label)) in objects.items():
			to = self.trackable_objects.get(objectID, None)
			if to is None:
				to = Trackable_Object(objectID, c)
				to.label = label
				to.color = self.mobile_net.clrs[to.label]
			else:
				to.centroids.append(c)
			self.trackable_objects[objectID] = to
			(start_x, start_y, end_x, end_y) = box
			cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), to.color, 1)
			overlay = frame.copy()
			radius = min( (end_x - start_x)//2 , (end_y - start_y)//2 )
			cv2.circle(overlay, (c[0], c[1]), (radius), to.color, -1)
			frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, 0)
			cv2.putText(frame, str(objectID), (c[0], c[1]), 0, 2, to.color, 3)
		return frame


	def read_video(self, vid_path, output, play):
		vs = Video_Thread(vid_path).start()
		fps = vs.fps()
		writer = None
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

			if output:
				if writer is None:
					fourcc = cv2.VideoWriter_fourcc(*"MJPG")
					writer = cv2.VideoWriter(
						output, fourcc, fps, (self.w, self.h), True)
				writer.write(frame)

			if play:
				cv2.imshow("Video", frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					vs.release()
					vs.stop()
					break

		# End while true reading video frames
	# End read_video()



##