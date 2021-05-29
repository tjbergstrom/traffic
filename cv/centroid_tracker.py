# centroid_tracker.py
# December 2020
# Tracking the center xy coordinates of detected objects.


from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class Centroid_Tracker:
	def __init__(self, max_disappear=32, max_distance=48):
		# A unique ID to keep track of the objects.
		self.next_objectID = 0
		# A list of objects stored by object ID.
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		# If an object is not in a frame, how many frames to keep looking for it.
		self.max_disappear = max_disappear
		# How close to consider a previous and current centeroid to be the same object.
		self.max_distance = max_distance


	def register(self, centroid, box, label):
		# Store the new centeroid in the list of objects, with the next object ID.
		self.objects[self.next_objectID] = (centroid, box, label)
		self.disappeared[self.next_objectID] = 0
		self.next_objectID += 1


	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]


	def update(self, boxs):
		# A quick check if there was nothing this frame.
		if len(boxs) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.max_disappear:
					self.deregister(objectID)
			return self.objects
		# An empty list for all of the centeroids in this frame.
		input_centroids = np.zeros((len(boxs), 2), dtype="int")
		input_boxs = [(0, 0, 0, 0)] * len(boxs)
		input_labels = [""] * len(boxs)
		# Check out each of the boxes from this frame.
		for (i, (start_x, start_y, end_x, end_y, label)) in enumerate(boxs):
			# Calculate the center coordinates from the box and add to the list.
			c_x = int((start_x + end_x) / 2.0)
			c_y = int((start_y + end_y) / 2.0)
			input_centroids[i] = (c_x, c_y)
			input_boxs[i] = (start_x, start_y, end_x, end_y)
			input_labels[i] = label
		# Quick check, if there aren't any tracking objects then just register everything.
		if len(self.objects) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i], input_boxs[i], input_labels[i])
		# Which centeroids correspond to which tracked objects.
		else:
			objectIDs = list(self.objects.keys())
			vals = list(self.objects.values())
			prev_centeroids = [x[0] for x in vals]
			boxs = [x[1] for x in vals]
			# Calculate the distance between each pair of new and previous centeroids.
			# D will be an n x n matrix.
			D = dist.cdist(np.array(prev_centeroids), input_centroids)
			# Sort the rows and columns by minimum distances.
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			used_rows = set()
			used_cols = set()
			# Check out each row, col index
			for (row, col) in zip(rows, cols):
				if row in used_rows or col in used_cols:
					continue
				if D[row, col] > self.max_distance:
					continue
				# Matched an object ID with the row.
				objectID = objectIDs[row]
				self.objects[objectID] = (input_centroids[col], input_boxs[col], input_labels[col])
				self.disappeared[objectID] = 0
				used_rows.add(row)
				used_cols.add(col)
			unused_rows = set(range(0, D.shape[0])).difference(used_rows)
			unused_cols = set(range(0, D.shape[1])).difference(used_cols)
			# If any centeroids have not been matched yet.
			if D.shape[0] >= D.shape[1]:
				# Check if any tracked objects have disappeared in this frame.
				for row in unused_rows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.max_disappear:
						self.deregister(objectID)
			else:
				# Or add new tracked objects for the left over centeroids.
				for col in unused_cols:
					self.register(input_centroids[col], input_boxs[i], input_labels[i])
		return self.objects



##
