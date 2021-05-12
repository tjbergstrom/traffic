# cviz.py
# May 2021
# Some extra and general video processing utilities that don't really belong anywhere else.


import os
import sys
import cv2


def vid_dimz(vid, width=None):
	cap = cv2.VideoCapture(vid)
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



##
