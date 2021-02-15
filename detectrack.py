# detectrack.py
# Feb 2021
# Multi-processed detection with tracking!
#
# Pass every n-th frame to the object detector to get vehicle etc detections,
# Use object tracking on the frames in between to keep track of only their locations,
# Also use multi-processing to process blocks of frames simultaneously.
# Implemented with both/either the MobileNet or the YoloV3 detectors.
#
# $ python3 detectrack.py -i vid_inputs/vid9.mp4 -o vid_outputs/tmp.avi -w 500 -f 10
#
# Use -y 100 to use the YoloV3 every 100th frame. Default is to not use it at all.
# Use -y == -f to use the YoloV3 always instead of the MobileNet


from traffyc.centroid_tracker import Centroid_Tracker
from traffyc.trackable_object import Trackable_Object
from traffyc.videostream import Video_Thread
from traffyc.mobilenet import Mnet
import numpy as np
import imutils
import dlib
import time
import cv2
import os
import sys
import argparse
import subprocess as sp
import multiprocessing as mp
from traffyc.yolo_proc import Yolo_Detection


class Traffic_Detection:
	def __init__(self, w, h, yolo, freq=5):
		self.w = w
		self.h = h
		self.trackers = []
		self.frame_count = 0
		self.yolo = yolo
		self.detect_freq = freq
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 64)
		self.YD = Yolo_Detection(w, h) if yolo>0 else None


	def yolo_detect(self, frame, rgb):
		self.trackers = []
		blob = cv2.dnn.blobFromImage(
			frame,
			1 / 255.0,
			(416, 416),
			swapRB=True,
			crop=False)
		self.YD.net.setInput(blob)
		output_layers = self.YD.net.forward(self.YD.ln)
		boxes = []
		confidences = []
		class_idxs = []
		for layer in output_layers:
			for detection in layer:
				scores = detection[5:]
				class_idx = np.argmax(scores)
				confidence = scores[class_idx]
				if confidence < 0.5:
					continue
				if class_idx not in self.YD.traffic_idxs:
					continue
				box = detection[0:4] * np.array([self.w, self.h, self.w, self.h])
				(center_x, center_y, width, height) = box.astype("int")
				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_idxs.append(class_idx)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				tracker = dlib.correlation_tracker()
				tracker.start_track(rgb, dlib.rectangle(x, y, x+w, y+h))
				self.trackers.append((tracker, self.YD.all_classes[class_idxs[i]]))


	def mobilenet_detect(self, frame, rgb, v_only=True):
		self.trackers = []
		blob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		MN.net.setInput(blob)
		detections = MN.net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence < 0.4:
				continue
			class_idx = int(detections[0, 0, i, 1])
			if class_idx not in MN.traffic_idxs:
				continue
			if v_only and class_idx not in MN.vehicles_only:
				continue
			if class_idx not in MN.vehicles_only and confidence < 0.6:
				continue
			box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			tracker = dlib.correlation_tracker()
			tracker.start_track(rgb, dlib.rectangle(start_x, start_y, end_x, end_y))
			self.trackers.append((tracker, MN.all_classes[class_idx]))


	def track(self, tracker, label, boxs, rgb):
		tracker.update(rgb)
		pos = tracker.get_position()
		start_x, start_y = int(pos.left()), int(pos.top())
		end_x, end_y = int(pos.right()), int(pos.bottom())
		boxs.append((start_x, start_y, end_x, end_y, label))
		return boxs


	def traffic_detections(self, frame, rgb):
		boxs = []
		if self.frame_count % self.yolo == 0 and self.yolo > 0:
			self.yolo_detect(frame, rgb)
			verbose(f"Yolo detection (frame {self.frame_count})")
		elif self.frame_count % self.detect_freq == 0:
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
				to.color = MN.colors[to.label]
				#to.color = self.YD.colrs[to.label]
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
			cv2.putText(frame, to.label, (start_x,start_y ), 0, 0.5, to.color, 2)
			#cv2.putText(frame, f"{objectID}", (c[0], c[1]), 0, 1.5, to.color, 3)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
		return frame


def read_video(proc_num):
	verbose(f"Process: {proc_num}, start frame {jump_unit*proc_num}/{frames}")
	TD = Traffic_Detection(w, h, yolo, freq)
	vs = cv2.VideoCapture(in_vid)
	first_block = True if proc_num==0 else False
	start_frame = jump_unit * proc_num
	if not first_block:
		start_frame -= 1
	vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	writer = cv2.VideoWriter(f"mpt/tmp_{proc_num}.mp4", fourcc, fps, (w, h), True)
	while TD.frame_count < jump_unit:
		check, frame = vs.read()
		if not check or frame is None:
		    break
		frame = imutils.resize(frame, width=w)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = TD.traffic_detections(frame, rgb)
		cv2.putText(frame, f"{ (jump_unit * proc_num) + TD.frame_count }", (10, 10), 0, 0.35, (20,255,10), 1)
		TD.frame_count += 1
		if TD.frame_count == (jump_unit // 2):
			verbose(f"Process {proc_num} 50% complete")
		if not first_block and TD.frame_count == 1:
			continue
		writer.write(frame)
	vs.release()
	writer.release()
	verbose(f"Process {proc_num} 100% complete")


def recombine_frames():
	verbose("Recombining frames...")
	tmp_files = [f"mpt/tmp_{i}.mp4" for i in range(processes)]
	f = open("tmps.txt", "w")
	for i in tmp_files:
		f.write(f"file {i} \n")
	f.close()
	cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmps.txt -vcodec copy {out_vid}"
	sp.Popen(cmd, shell=True).wait()
	for i in tmp_files:
		os.remove(i)
	os.remove("tmps.txt")


def multi_process():
	p = mp.Pool(processes)
	p.map(read_video, range(processes))
	recombine_frames()


def meta_info(vid, width=None):
	cap = cv2.VideoCapture(vid)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if width is None:
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	else:
		(h, w) = imutils.resize(cap.read()[1], width=width).shape[:2]
	cap.release()
	return w, h, fps, frames


def checkargs(in_vid, out_vid, w, y, freq, frames, processes):
	if not os.path.isfile(in_vid):
		sys.exit(f"\'{in_vid}\' is not a filepath")
	if not os.path.isdir(os.path.dirname(out_vid)):
		sys.exit(f"Cannot save an output video to \'{out_vid}\'")
	if not os.path.basename(out_vid):
		sys.exit(f"No output file specified \'{out_vid}\'")
	if w < 360 or w > 1800:
		sys.exit(f"Width \'{w}\' out of range")
	if freq < 2 or freq > 40:
		sys.exit(f"Detection frequency \'{freq}\' not supported")
	if processes == 0:
		sys.exit(f"No processors found")
	if frames < processes:
		sys.exit(f"Video is too short")
	if y != -1 and y < freq:
		sys.exit(f"-y {y} and -f {freq} not supported")
	if os.path.isfile(out_vid):
		print(f"Warning: will be over-writing output video \'{out_vid}\'")
		time.sleep(3.0)


def verbose(msg):
	if args["verbose"]:
		print(msg)


if __name__ == "__main__":
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=True)
	ap.add_argument("-w", "--width", type=int, default=None)
	ap.add_argument("-f", "--freq", type=int, default=2)
	ap.add_argument("-v", "--verbose", type=bool, default=True)
	ap.add_argument("-y", "--yolo", type=int, default=-1, help="how often to use YoloV3")
	args = vars(ap.parse_args())
	out_vid = args["output"]
	in_vid = args["input"]
	yolo = args["yolo"]
	freq = args["freq"]
	w, h, fps, frames = meta_info(in_vid, args["width"])
	fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
	processes = min(mp.cpu_count(), frames)
	checkargs(in_vid, out_vid, w, yolo, freq, frames, processes)
	jump_unit = frames // processes
	#YD = Yolo_Detection(w, h)
	MN = Mnet()
	os.system("mkdir -p mpt")
	multi_process()
	if os.path.isfile(out_vid):
		print(f"Output video successfully saved")
	else:
		sys.exit(f"Output video not saved")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
