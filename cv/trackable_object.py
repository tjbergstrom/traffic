# trackable_object.py
# December 2020
# Storing the center coordinates locations and other info for tracked objects.


class Trackable_Object:
	def __init__(self, objectID, centroid):
		self.objectID = objectID
		self.centroids = [centroid]
		# You don't actually need to store a MobileNet color here.
		# But this would allow you to assign unique random colors to each new object.
		self.color = (0, 0, 0)
		self.label = ""


##
