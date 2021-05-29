# trackable_object.py
# December 2020
# Storing the center coordinates locations and other info for tracked objects.


class Trackable_Object:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]
		self.color = (0, 0, 0)
		self.label = ""


##
