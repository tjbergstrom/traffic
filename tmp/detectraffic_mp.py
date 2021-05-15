# detectraffic_mp.py
# May 2021
# Multi-processed traffic detection with object tracking.
#
# Pass every n-th frame to the object detector to get new vehicle etc detections,
# Use object tracking on the frames in between to keep track of only their locations,
# Also use multi-processing to process blocks of frames simultaneously.
#
# $ python3 detectraffic_mp.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -w 500 -f 10


import cviz
import cv2
import os
import sys
import math
import time
import argparse
import subprocess as sp
import multiprocessing as mp
from detectrack import Traffic_Detection


def read_video_mp(proc_num):
	TD = Traffic_Detection(w, h, freq, jump_unit)
	vs = cv2.VideoCapture(in_vid)
	start_frame = jump_unit * proc_num
	first_block = (proc_num == 0)
	if not first_block:
		start_frame -= 1
		TD.jump_unit += 1
	vs_pos = start_frame
	vs = cviz.set_pos(vs, start_frame)
	writer = cviz.vid_writer(f"mpt/tmp_{proc_num}.avi", w, h, fps)
	while TD.frame_count < TD.jump_unit:
		check, frame = vs.read()
		if not check or frame is None:
		    break
		if resize_w:
			frame = cviz.resize(frame, width=TD.w)
		frame = TD.traffic_detections(frame)
		TD.frame_count += 1
		vs_pos += 1
		if not first_block and TD.frame_count == 1:
			continue
		cv2.putText(frame, f"{vs_pos}", (10,25), 0, 0.35, (20,255,10), 1)
		writer.write(frame)
		assert vs_pos == int(vs.get(cv2.CAP_PROP_POS_FRAMES)), "Frame position is off"
		if (proc_num == processes // 2) and (TD.frame_count == TD.jump_unit // 2):
			print(f"50% complete")
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
	p.map(read_video_mp, range(processes))
	recombine_frames()


if __name__ == "__main__":
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="input video filepath")
	ap.add_argument("-o", "--output", required=True, help="save output video filepath")
	ap.add_argument("-w", "--width", type=int, default=None, help="resize video frame width")
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
		sys.exit(f"Not a valid input video type, '{in_vid}'")
	if resize_w and (resize_w < 180 or resize_w > 1800):
		sys.exit(f"Width '{resize_w}' out of range")
	if freq < 2 or freq > 40:
		sys.exit(f"Detection frequency '{freq}' not supported")
	if os.path.isfile(out_vid):
		os.remove(out_vid)

	w, h = cviz.vid_dimz(in_vid, resize_w)
	fps = cviz.vid_fps(in_vid)
	frames = cviz.frame_cnt(in_vid)
	processes = min(mp.cpu_count(), frames)
	jump_unit = math.ceil(frames / processes)

	if processes == 0:
		sys.exit(f"No processors found")
	if frames <= 0:
		sys.exit(f"Error with video frames count")

	print(f"Processing {frames} frames on {processes} processors... ")
	multi_process()

	if os.path.isfile(out_vid):
		if not frames == cviz.frame_cnt(out_vid):
			print("Saved incorrectly, frame count off")
		if not fps == cviz.vid_fps(out_vid):
			print("Saved incorrectly, fps is off")
	else:
		sys.exit(f"Output video not saved")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
