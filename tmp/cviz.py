# cviz.py
# May 2021
# Some extra and general video processing utilities that don't really belong anywhere else.


import os
import sys
import cv2


def vid_dimz(src, width=None):
	cap = cv2.VideoCapture(src)
	if width is None:
	    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	else:
	    (h, w) = resize(cap.read()[1], width=width).shape[:2]
	cap.release()
	return w, h


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
	ext = os.path.splitext(in_vid)[1]
	if ext in exts:
		return 1
	return 0


def frame_cnt(src, manual=False):
	frames = 0
	cap = cv2.VideoCapture(src)
	try:
		frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	except:
		frames = 0
	if frames <= 0 or manual:
		frames = 0
		while True:
			check, frame = cap.read()
			if not check or frame is None:
				break
			frames += 1
	cap.release()
	return frames


def vid_fps(src):
	cap = cv2.VideoCapture(src)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	cap.release()
	return fps


def set_pos(vs, start_frame):
	vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	while vs.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
		check, frame = vs.read()
		if not check or frame is None:
		    break
	return vs



##
