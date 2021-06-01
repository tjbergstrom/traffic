# mobilnet.py
# November 2020
#
# Set up the MobileNet, read from a file location.
# And set up the classes that it detects.
# Specify the traffic related objects and set up colors for them.


import cv2


mndir = "detectors/MobileNet"

class Mnet:
	def __init__(self):
		self.weights = f"{mndir}/MobileNetSSD_deploy.prototxt.txt"
		self.model = f"{mndir}/MobileNetSSD_deploy.caffemodel"
		self.net = cv2.dnn.readNetFromCaffe(self.weights, self.model)
		self.traffic_idxs = [2, 6, 7, 14, 15]
		self.vehicles_only = [6, 7]
		self.all_classes = [
			"background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"
		]
		self.colors = {
			"bicycle" 	: (0, 255, 0),
			"person" 	: (0, 0, 255),
			"car" 		: (255, 0, 0),
			"bus" 		: (225, 100, 0),
			"motorbike"	: (125, 255, 0),
		}
		self.colrs = [
			(0, 255, 0), (0, 0, 255), (255, 0, 0), (225, 100, 0), (125, 255, 0),
			(102, 0, 102), (255, 0, 102), (0, 255, 255), (255, 51, 0), (255, 255, 0),
		]



##
