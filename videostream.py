# videostream.py
# November 2020
# Starts a thread for an input video file
# Adds video frames to a thread-safe Python Queue
# For faster video stream processing


from threading import Thread
from queue import Queue
import cv2
import time


class Video_Thread:
	def __init__(self, vid):
		self.stream = cv2.VideoCapture(vid)
		self.quit = 0
		self.que = Queue(maxsize=256)
		self.thread = Thread(
			target=self.update,
			args=(),
			daemon=True
		)
		time.sleep(1.0)

	def start(self):
		self.thread.start()
		return self

	def update(self):
		while 1:
			if self.quit:
				break
			if not self.que.full():
				check, frame = self.stream.read()
				if not check:
					self.quit = 1
				self.que.put(frame)
			else:
				time.sleep(0.1)
		self.stream.release()

	def read(self):
		return self.que.get()

	def more_frames(self, t=0):
		while self.que.qsize()==0 and not self.stopped and t<5:
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



##
