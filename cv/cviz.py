# cviz.py
# May 2021
# Some video processing utilities that are nice to have.


import os
import sys
import cv2


# Get the original height x width dimensions of a video.
# Or get the new dimensions after resizing to a specified width.
def vid_dimz(src, width=None):
	cap = cv2.VideoCapture(src)
	if width is None:
	    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	else:
	    (h, w) = resize(cap.read()[1], width=width).shape[:2]
	cap.release()
	return w, h


# Resize a video frame to a new width, preserving the original aspect ratio.
def resize(frame, width=None, height=None):
	if width is None and height is None:
	    return frame
	(h, w) = frame.shape[:2]
	if width is None:
	    ratio = height / float(h)
	    dim = (int(w * ratio), height)
	else:
	    ratio = width / float(w)
	    dim = (width, int(h * ratio))
	return cv2.resize(frame, dim)


def vid_writer(output, w, h, fps):
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(output, fourcc, fps, (w, h), True)
	return writer


def valid_vidtyp(in_vid):
	exts = set(['.avi', '.mp4', '.mjpeg',])
	ext = os.path.splitext(os.path.basename(in_vid))[1]
	if ext in exts:
		return 1
	return 0


# Get the number of frames in a video.
# If the video property fails, count the frames manually, O(n) time.
def frame_cnt(src, manual=False):
	cap = cv2.VideoCapture(src)
	frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if frames <= 0 or manual:
		frames = 0
		while True:
			check, frame = cap.read()
			if not check or frame is None:
				break
			frames += 1
	cap.release()
	return frames


# fps = frames per second
def vid_fps(src):
	cap = cv2.VideoCapture(src)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	cap.release()
	return fps


# Set the index of the frame to be read next in a video stream,
# Useful for setting the first frame for a block of frames.
# If the video property fails, set it manually, O(n) time.
def set_pos(vs, start_frame):
	vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	while vs.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
		check, frame = vs.read()
		if not check or frame is None:
		    break
	return vs


# Convert a video to .avi, O(n) time.
def avi_conv(src):
	path, ext = os.path.splitext(src)
	cap = cv2.VideoCapture(src)
	w, h = vid_dimz(src)
	writer = vid_writer(f"{path}.avi", w, h, int(cap.get(cv2.CAP_PROP_FPS)))
	while True:
		check, frame = cap.read()
		if not check or frame is None:
		    break
		writer.write(frame)
	cap.release()
	writer.release()
	return f"{path}.avi"



##
