# detectrack.py
# May 2021
# Multi-processed detection with tracking
#
# Pass every n-th frame to the object detector to get new vehicle etc detections,
# Use object tracking on the frames in between to keep track of only their locations,
# Also use multi-processing to process blocks of frames simultaneously.
#
# $ python3 detectrack.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -w 500 -f 10 -v tru


from centroid_tracker import Centroid_Tracker
from trackable_object import Trackable_Object
from mobilenet import Mnet
import numpy as np
import cviz
import dlib
import cv2
import os
import sys
import math
import time
import argparse
import subprocess as sp
import multiprocessing as mp


class Traffic_Detection:
	def __init__(self, w, h, jump_unit, freq=5):
		self.w = w
		self.h = h
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


	def traffic_detections(self, frame):
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
				to.color = MN.colors[to.label]
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
			cv2.putText(frame, to.label, (start_x+10,start_y+15), 0, 0.5, to.color, 2)
			cv2.putText(frame, f"{objectID}", (c[0], c[1]), 0, 1.1, to.color, 3)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
		return frame


def read_video(proc_num):
	TD = Traffic_Detection(w, h, jump_unit, freq)
	vs = cv2.VideoCapture(in_vid)
	start_frame = jump_unit * proc_num
	if not (proc_num == 0):
		start_frame -= 1
		TD.jump_unit += 1
	vs = cviz.set_pos(vs, start_frame)
	writer = cviz.vid_writer(f"mpt/tmp_{proc_num}.avi", w, h, fps)
	while TD.frame_count < TD.jump_unit:
		check, frame = vs.read()
		if not check or frame is None:
		    break
		if resize_w:
			frame = cviz.resize(frame, width=w)
		frame = TD.traffic_detections(frame)
		TD.frame_count += 1
		if not (proc_num == 0) and TD.frame_count == 1:
			continue
		if (proc_num == processes // 2) and TD.frame_count == (jump_unit // 2):
			verbose(f"50% complete")
		vs_pos = (jump_unit * proc_num) + TD.frame_count
		if not (proc_num == 0):
			vs_pos -= 1
		cv2.putText(frame, f"{vs_pos}", (w-25,10), 0, 0.35, (20,255,10), 1)
		assert vs_pos == int(vs.get(cv2.CAP_PROP_POS_FRAMES)), "Frame position is off"
		writer.write(frame)
	vs.release()
	writer.release()


def recombine_frames():
	tmp_files = [f"mpt/tmp_{i}.avi" for i in range(processes)]
	with open("tmps.txt", "w") as f:
		for i in tmp_files:
			f.write(f"file {i} \n")
	cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmps.txt -vcodec copy {out_vid}"
	sp.Popen(cmd, shell=True).wait()
	os.remove("tmps.txt")
	os.system("rm -f -r mpt")


def multi_process():
	os.system("rm -f -r mpt")
	os.system("mkdir -p mpt")
	p = mp.Pool(processes)
	p.map(read_video, range(processes))
	recombine_frames()


def verbose(msg):
	if args["verbose"]:
		print(msg)


if __name__ == "__main__":
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=True)
	ap.add_argument("-w", "--width", type=int, default=None)
	ap.add_argument("-v", "--verbose", type=bool, default=False)
	ap.add_argument("-f", "--freq", type=int, default=5, help="frequency of detections")
	args = vars(ap.parse_args())

	resize_w = args["width"]
	out_vid = args["output"]
	in_vid = args["input"]
	freq = args["freq"]

	if not os.path.isfile(in_vid):
		sys.exit(f"'{in_vid}' is not a filepath")
	if not os.path.isdir(os.path.dirname(out_vid)):
		sys.exit(f"Cannot save an output video to '{out_vid}'")
	if not os.path.basename(out_vid):
		sys.exit(f"No output file specified '{out_vid}'")
	if not cviz.valid_vidtyp(in_vid):
		sys.exit(f"Not a valid video type, '{in_vid}'")
	if resize_w and (resize_w < 360 or resize_w > 1800):
		sys.exit(f"Width '{w}' out of range")
	if freq < 2 or freq > 40:
		sys.exit(f"Detection frequency '{freq}' not supported")
	if os.path.isfile(out_vid):
		os.remove(out_vid)
	#if os.path.splitext(os.path.basename(in_vid))[1] == '.mjpeg':
		#in_vid = cviz.avi_conv(in_vid)

	w, h = cviz.vid_dimz(in_vid, resize_w)
	frames = cviz.frame_cnt(in_vid)
	fps = cviz.vid_fps(in_vid)
	processes = min(mp.cpu_count(), frames)
	jump_unit = math.ceil(frames / processes)
	MN = Mnet()

	if processes == 0:
		sys.exit(f"No processors found")
	if frames <= 0:
		sys.exit(f"Error with video frames count")

	verbose(f"Processing {frames} frames on {processes} processors")
	multi_process()

	if os.path.isfile(out_vid):
		if frames != cviz.frame_cnt(out_vid):
			sys.exit(f"Output video not correctly saved (frame count off)")
		if fps != cviz.vid_fps(out_vid):
			sys.exit(f"Output video not correctly saved (fps off)")
	else:
		sys.exit(f"Output video not saved")
	verbose(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
