# trackable_object.py
# December 2020
# Software Engineering Project


class Trackable_Object:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]
		self.color = (0, 0, 0)
		self.label = ""


##
