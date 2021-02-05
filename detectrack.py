


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
import os
import sys
import cv2
import time
import imutils
import argparse
import subprocess as sp
import multiprocessing as mp
from traffyc.yolo_proc import Yolo_Detection




class Traffic_Detection:
	def __init__(self, w, h, freq=5):
		self.w = w
		self.h = h
		self.trackers = []
		self.frame_count = 0
		self.detect_freq = freq
		self.mobile_net = Mnet()
		self.net = self.mobile_net.net
		self.trackable_objects = {}
		self.ct = Centroid_Tracker(8, 32)


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
			box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			tracker = dlib.correlation_tracker()
			tracker.start_track(rgb,  dlib.rectangle(start_x, start_y, end_x, end_y))
			self.trackers.append((tracker, self.mobile_net.all_classes[class_idx]))


	def track(self, tracker, label, boxs, rgb):
		tracker.update(rgb)
		pos = tracker.get_position()
		start_x, start_y = int(pos.left()), int(pos.top())
		end_x, end_y = int(pos.right()), int(pos.bottom())
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
				to.color = self.mobile_net.colors[to.label]
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
			cv2.putText(frame, str(objectID), (c[0], c[1]), 0, 1.5, to.color, 3)
			cv2.putText(frame, f"({c[0]},{c[1]})", (c[0]-radius//2, c[1]), 0, 0.35, (255,255,255), 1)
		return frame


def read_video(proc_num):
    verbose(f"Process: {proc_num}, start frame {jump_unit*proc_num}/{frames}")
    TD = Traffic_Detection(w, h, freq)
    vs = cv2.VideoCapture(in_vid)
    vs.set(cv2.CAP_PROP_POS_FRAMES, jump_unit * proc_num)
    proc_frames = 0
    writer = cv2.VideoWriter(f"tmp_{proc_num}.mp4", fourcc, fps, (w, h), True)
    while proc_frames < jump_unit:
        check, frame = vs.read()
        if not check or frame is None:
            break
        frame = imutils.resize(frame, width=w)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TD.traffic_detections(frame, rgb)
        writer.write(frame)
        #writer.write(YD.detect(imutils.resize(frame, w)))
        proc_frames += 1
        TD.frame_count += 1
        if proc_frames == (jump_unit // 2):
            verbose(f"Process {proc_num} 50% complete")
    vs.release()
    writer.release()
    verbose(f"Process {proc_num} 100% complete")


def recombine_frames():
    verbose("Recombining frames...")
    tmp_files = [f"tmp_{i}.mp4" for i in range(processes)]
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


def verbose(msg):
    if v:
        print(msg)


def checkargs(in_vid, out_vid, w, freq, frames, processes):
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
	if os.path.isfile(out_vid):
		print(f"Warning: will be over-writing output video \'{out_vid}\'")
		time.sleep(3.0)


if __name__ == "__main__":
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=False)
	ap.add_argument("-w", "--width", type=int, default=None)
	ap.add_argument("-f", "--freq", type=int, default=2)
	ap.add_argument("-v", "--verbose", type=bool, default=True)
	args = vars(ap.parse_args())
	in_vid = args["input"]
	out_vid = args["output"]
	width = args["width"]
	freq = args["freq"]
	v = args["verbose"]
	w, h, fps, frames = meta_info(in_vid, )
	fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
	processes = min(mp.cpu_count(), frames)
	checkargs(in_vid, out_vid, w, freq, frames, processes)
	jump_unit = frames // processes
	#YD = Yolo_Detection(w, h)
	multi_process()
	if os.path.isfile(out_vid):
		print(f"Output video successfully saved")
	else:
		sys.exit(f"Output video not saved")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
