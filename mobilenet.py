# mobilnet.py
# November 2020
# Some properties of MobileNet that are used to make detections


import cv2


class Mnet:
	def __init__(self):
		self.weights = "MobileNet/MobileNetSSD_deploy.prototxt.txt"
		self.model = "MobileNet/MobileNetSSD_deploy.caffemodel"
		self.net = cv2.dnn.readNetFromCaffe(self.weights, self.model)
		self.traffic_idxs = [2, 6, 7, 14, 15]
		self.all_classes = [
			"background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"
		]
		self.colors = {
			"person" : (0,0,255),
			"car" : (255,0,0),
			"bus" : (225,25,0),
			"motorbike" : (128,255,0),
			"bicycle" : (0,255,0)
		}



##
