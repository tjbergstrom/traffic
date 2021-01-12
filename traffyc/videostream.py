# videostream.py
# November 2020
# Starts a thread for an video file and adds frames to a Queue for faster processing.


from threading import Thread
from queue import Queue
import imutils
import cv2
import time


class Video_Thread:
	def __init__(self, src, frames):
		self.stream = cv2.VideoCapture(src)
		self.que = Queue(maxsize=256)
		self.quit = 0
		self.thread = Thread(
			target=self.update,
			args=(),
			daemon=True
		)
		self.statuses = {
			int(frames*0.25) : "25% complete",
			int(frames*0.50) : "50% complete",
			int(frames*0.75) : "75% complete",
		}


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
		self.stream.release()


	def fps(self):
		return self.stream.get(cv2.CAP_PROP_FPS)


	def vid_writer(self, output, w, h):
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, self.fps(), (w, h), True)
		return writer


	@staticmethod
	def vid_dims(vid, width):
		cap = cv2.VideoCapture(vid)
		frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		(h, w) = imutils.resize(cap.read()[1], width=width).shape[:2]
		cap.release()
		return w, h, frames


	def status(self, frame_count):
		if frame_count in self.statuses:
			print(self.statuses.get(frame_count))



##
