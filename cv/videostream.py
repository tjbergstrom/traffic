# videostream.py
# November 2020
# Start a thread for reading video and add frames to a Queue for faster processing.


from threading import Thread
from queue import Queue
import cviz
import cv2
import time


class Video_Thread:
	def __init__(self, src):
		self.stream = cv2.VideoCapture(src)
		self.que = Queue(maxsize=256)
		self.src = src
		self.quit = 0
		self.thread = Thread(
			target=self.update,
			args=(),
			daemon=True,
		)


	def start(self):
		self.thread.start()
		return self


	def update(self):
		while True:
			if self.quit:
				break
			if not self.que.full():
				check, frame = self.stream.read()
				if not check or frame is None:
					self.quit = 1
					break
				self.que.put(frame)
			else:
				time.sleep(0.1)
		self.release()


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
		self.quit = 1
		self.stream.release()


	def fps(self):
		return self.stream.get(cv2.CAP_PROP_FPS)


	def frames(self):
		frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
		if frames <= 0:
			frames = cviz.frame_cnt(self.src)
		return frames


	def set_pos(self, start_frame):
		return cviz.set_pos(self.stream, start_frame)



##