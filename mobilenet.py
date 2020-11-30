# mobilnet.py
# November 2020
# Some properties of MobileNet that are used to make detections


import cv2


class Mnet:
	def __init__(self):
		self.weights = "MobileNet/MobileNetSSD_deploy.prototxt.txt"
		self.model = "MobileNet/MobileNetSSD_deploy.caffemodel"
		self.net = cv2.dnn.readNetFromCaffe(self.weights, self.model)
		self.all_classes = [
			"background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"
		]
		self.traffic_idxs = [2, 6, 7, 14, 15]
		self.colors = [(0,0,0)] * len(self.all_classes)
		self.colors[2] = (0,255,0) # bicycle
		self.colors[15] = (0,0,255) # person
		self.colors[14] = (128,255,0) # motorcycle
		self.colors[6] = self.colors[7] = (255,0,0) # car, bus



##
