# detectraffic.py
# May 2021
# Traffic detection with object tracking to increase efficiency.
#
# Pass every n-th frame to the object detector to get new vehicle etc detections,
# Use object tracking on the frames in between to keep track of only their locations.
#
# $ python3 detectraffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -w 500 -f 10


from detectrack import Traffic_Detection
from videostream import Video_Thread
import argparse
import cviz
import time
import cv2
import sys
import os


def read_video(vid_path, output, resize_w, freq):
	w, h = cviz.vid_dimz(vid_path, resize_w)
	TD = Traffic_Detection(w, h, freq)
	vs = Video_Thread(vid_path).start()
	writer = cviz.vid_writer(output, w, h, vs.fps())
	frames = vs.frames()
	print(f"Processing {frames} frames... ")
	while True:
		frame = vs.read()
		if frame is None:
			break
		if resize_w:
			frame = cviz.resize(frame, width=w)
		frame = TD.traffic_detections(frame, obj_id=True)
		cv2.putText(frame, f"{TD.frame_count}", (10,25), 0, 0.35, (20,255,10), 1)
		writer.write(frame)
		TD.frame_count += 1
		if TD.frame_count == frames // 2:
			print(f"50% complete")
	vs.release()
	writer.release()


if __name__ == '__main__':
	start = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="input video filepath")
	ap.add_argument("-o", "--output", required=True, help="save output video filepath")
	ap.add_argument("-w", "--width", type=int, default=None, help="resize video frame width")
	ap.add_argument("-f", "--freq", type=int, default=5, help="detection frequency, default is every 5 frames")
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
	if freq < 2 or freq > 20:
		sys.exit(f"Detection frequency '{freq}' not supported")
	if os.path.isfile(out_vid):
		os.remove(out_vid)

	read_video(in_vid, out_vid, resize_w, freq)

	if os.path.isfile(out_vid):
		if not cviz.frame_cnt(in_vid) == cviz.frame_cnt(out_vid):
			sys.exit("Saved incorrectly, frame count off")
		if not cviz.frame_cnt(in_vid) == cviz.frame_cnt(out_vid):
			sys.exit("Saved incorrectly, fps is off")
	else:
		sys.exit(f"Output video not saved")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")



##
