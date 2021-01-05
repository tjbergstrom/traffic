# videostream.py
# November 2020
# Starts a thread for an input video file
# Adds video frames to a thread-safe Python Queue
# For faster video stream processing


from threading import Thread
from queue import Queue
import imutils
import cv2
import time


class Video_Thread:
	def __init__(self, src):
		self.stream = cv2.VideoCapture(src)
		self.que = Queue(maxsize=256)
		self.quit = 0
		self.thread = Thread(
			target=self.update,
			args=(),
			daemon=True
		)


	def start(self):
		time.sleep(0.5)
		self.thread.start()
		return self


	def update(self):
		while True:
			if self.quit:
				break
			if not self.que.full():
				check, frame = self.stream.read()
				if not check:
					self.quit = 1
					break
				self.que.put(frame)
			else:
				time.sleep(0.1)
		self.stream.release()


	def read(self):
		if not self.more_frames():
			return None
		return self.que.get()


	def more_frames(self, t=0):
		while self.que.qsize()==0 and not self.quit and t<5:
			time.sleep(0.1)
			t += 1
		return self.que.qsize() > 0


	def stop(self):
		self.quit = 1
		self.thread.join()


	def release(self):
		self.stream.release()


	def fps(self):
		return self.stream.get(cv2.CAP_PROP_FPS)


	def vid_writer(self, output, w, h):
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, self.fps(), (w, h), True)
		return writer


	def vid_dims(vid, width):
		cap = cv2.VideoCapture(vid)
		frame = cap.read()[1]
		frame = imutils.resize(frame, width=width)
		(h, w) = frame.shape[:2]
		cap.release()
		return w, h



##
