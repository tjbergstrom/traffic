# centroid_tracker.py
# December 2020


from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class Centroid_Tracker:
	def __init__(self, max_disappear=32, max_distance=48):
		self.next_objectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.max_disappear = max_disappear
		self.max_distance = max_distance


	def register(self, centroid, box, label):
		self.objects[self.next_objectID] = (centroid, box, label)
		self.disappeared[self.next_objectID] = 0
		self.next_objectID += 1


	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]


	def update(self, boxs):
		if len(boxs) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.max_disappear:
					self.deregister(objectID)
			return self.objects
		input_centroids = np.zeros((len(boxs), 2), dtype="int")
		input_boxs = [(0, 0, 0, 0)] * len(boxs)
		input_labels = [""] * len(boxs)
		for (i, (start_x, start_y, end_x, end_y, label)) in enumerate(boxs):
			c_x = int((start_x + end_x) / 2.0)
			c_y = int((start_y + end_y) / 2.0)
			input_centroids[i] = (c_x, c_y)
			input_boxs[i] = (start_x, start_y, end_x, end_y)
			input_labels[i] = label
		if len(self.objects) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i], input_boxs[i], input_labels[i])
		else:
			objectIDs = list(self.objects.keys())
			vals = list(self.objects.values())
			object_centroids = [x[0] for x in vals]
			boxs = [x[1] for x in vals]
			D = dist.cdist(np.array(object_centroids), input_centroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			used_rows = set()
			used_cols = set()
			for (row, col) in zip(rows, cols):
				if row in used_rows or col in used_cols:
					continue
				if D[row, col] > self.max_distance:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = (input_centroids[col], input_boxs[col], input_labels[col])
				self.disappeared[objectID] = 0
				used_rows.add(row)
				used_cols.add(col)
			unused_rows = set(range(0, D.shape[0])).difference(used_rows)
			unused_cols = set(range(0, D.shape[1])).difference(used_cols)
			if D.shape[0] >= D.shape[1]:
				for row in unused_rows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.max_disappear:
						self.deregister(objectID)
			else:
				for col in unused_cols:
					self.register(input_centroids[col], input_boxs[i], input_labels[i])
		return self.objects



##
