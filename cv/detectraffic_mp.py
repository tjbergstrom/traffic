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


# Process a block of frames.
def read_video_mp(proc_num):
	# proc_num is which process this is, which determines which block of frames to read.
	w, h = cviz.vid_dimz(in_vid, resize_w)
	TD = Traffic_Detection(w, h, freq)
	vs = cv2.VideoCapture(in_vid)
	# Set the video stream to start reading from the first frame of this block of frames.
	frame_block = jump_unit
	start_frame = jump_unit * proc_num
	first_block = (proc_num == 0)
	if not first_block:
		start_frame -= 1
		frame_block += 1
	vs_pos = start_frame
	vs = cviz.set_pos(vs, start_frame)
	# Saving the block of frames as a temporary video.
	writer = cviz.vid_writer(f"mpt/tmp_{proc_num}.avi", w, h, cviz.vid_fps(in_vid))
	# Read all of the video frames in this block.
	while TD.frame_count < frame_block:
		check, frame = vs.read()
		if not check or frame is None:
		    break
		if resize_w:
			frame = cviz.resize(frame, width=w)
		# Do the traffic detection stuff, draw on the frames, etc.
		frame = TD.traffic_detections(frame, obj_id=track)
		TD.frame_count += 1
		vs_pos += 1
		if not first_block and TD.frame_count == 1:
			continue
		cv2.putText(frame, f"{vs_pos-1}", (10,25), 0, 0.35, (20,255,10), 1)
		# Save the output processed frame.
		writer.write(frame)
		assert vs_pos == int(vs.get(cv2.CAP_PROP_POS_FRAMES)), "Frame position is off"
		if (proc_num == processes // 2) and (TD.frame_count == frame_block // 2):
			print(f"50% complete")
	vs.release()
	writer.release()


def multi_process_vid():
	os.system("rm -f -r mpt")
	os.system("mkdir -p mpt")
	# Multi-process the blocks of frames (saving them as temporary videos).
	# read_video_mp is the multi-processed function.
	# The process number is the only argument that can be passed to it.
	p = mp.Pool(processes)
	p.map(read_video_mp, range(processes))
	# Recombine the blocks of frames (the temporary videos) into the finished video.
	tmp_files = [f"mpt/tmp_{i}.avi" for i in range(processes)]
	with open("tmps.txt", "w") as f:
		for i in tmp_files:
			f.write(f"file {i} \n")
	cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmps.txt -vcodec copy {out_vid}"
	sp.Popen(cmd, shell=True).wait()
	os.remove("tmps.txt")
	os.system("rm -f -r mpt")


if __name__ == "__main__":
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="input video filepath")
	ap.add_argument("-o", "--output", required=True, help="save output video filepath")
	ap.add_argument("-w", "--width", type=int, default=None, help="resize video frame width")
	ap.add_argument("-f", "--freq", type=int, default=5, help="detection frequency, default is every 5 frames")
	ap.add_argument("-t", "--track", type=bool, default=False, help="single processed for true object tracking")
	args = vars(ap.parse_args())

	resize_w = args["width"]
	out_vid = args["output"]
	in_vid = args["input"]
	track = args["track"]
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

	# How many frames are in this video.
	frames = cviz.frame_cnt(in_vid)
	if frames <= 0:
		sys.exit(f"Error with video frames count")
	# How many processors are on this machine.
	processes = mp.cpu_count()
	if processes == 0:
		sys.exit(f"No processors found")
	# If it's a short video and you have a lot of processors, just set it to 4 processes.
	if processes >= (frames // 4):
		processes = 4
	# To enable true object tracking, one process will make all frames read contiguously.
	if track:
		processes = 1
	# Divide frames by processors, that's how many frames each block needs to process.
	jump_unit = math.ceil(frames / processes)
	print(f"Processing {frames} frames on {processes} processors... ")
	multi_process_vid()

	if os.path.isfile(out_vid):
		if not frames == cviz.frame_cnt(out_vid):
			print("Saved incorrectly, frame count off")
		if not cviz.vid_fps(in_vid) == cviz.vid_fps(out_vid):
			print("Saved incorrectly, fps is off")
	else:
		sys.exit(f"Output video not saved")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
