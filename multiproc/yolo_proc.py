


import numpy as np
import argparse
import imutils
import time
import cv2
import os


class Yolo_Detection:
	def __init__(self, w, h):
		self.all_classes = open("yolo-coco/coco.names").read().strip().split("\n")
		self.traffic_idxs = [0, 1, 2, 3, 5, 7, 9, 11]
		self.traffic_labels = [self.all_classes[i] for i in self.traffic_idxs]
		self.colors = [(0,0,0)] * len(self.all_classes)
		self.colors[0] = (0,0,255) # person
		self.colors[1] = (0,255,0) # bike
		self.colors[3] = (128,255,0) # motorcycle
		self.colors[2] = self.colors[5] = self.colors[7] = (255,0,0) # car, bus, truck
		self.colors[9] = self.colors[11] = (0,0,0) # stop sign, traffic light
		self.weights = "yolo-coco/yolov3.weights"
		self.cfg = "yolo-coco/yolov3.cfg"
		self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
		self.ln = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		self.w = w
		self.h = h


	def detect(self, frame):
		blob = cv2.dnn.blobFromImage(
			frame,
			1 / 255.0,
			(416, 416),
			swapRB=True,
			crop=False
		)
		self.net.setInput(blob)
		output_layers = self.net.forward(self.ln)
		boxes = []
		confidences = []
		class_idxs = []
		for layer in output_layers:
			for detection in layer:
				scores = detection[5:]
				class_idx = np.argmax(scores)
				confidence = scores[class_idx]
				if confidence < 0.5:
					continue
				if class_idx not in self.traffic_idxs:
					continue
				box = detection[0:4] * np.array([self.w, self.h, self.w, self.h])
				(center_x, center_y, width, height) = box.astype("int")
				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_idxs.append(class_idx)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				color = [int(c) for c in self.colors[class_idxs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				label = self.all_classes[class_idxs[i]]
				cv2.putText(frame, label, (x, y - 5), 0, 0.5, color, 2)
		return frame



##
