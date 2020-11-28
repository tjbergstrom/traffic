


# python3 detect_traffic_mobilenet_vid.py -i vid_inputs/vid3.mp4 -o vid_outputs/1.avi

import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=False)
ap.add_argument("-v", "--playvideo", type=bool, default=False)
args = vars(ap.parse_args())

# Pick out the relevant MoblineNet classes (& assign colors for the bounding boxes)
all_classes = [
	"background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"
]
traffic_idxs = [2, 6, 7, 14, 15] # So we can ignore all other indexes during detection
traffic_classes = [all_classes[i] for i in traffic_idxs]
colors = [(0,0,0)] * len(all_classes)
colors[15] = (0,0,255) # person
colors[2] = (0,255,0) # bike
colors[14] = (128, 255, 0) # motorcycle
colors[6] = colors[7] = (255,0,0) # car, bus

print("-Loading model")
weights = "MobileNet/MobileNetSSD_deploy.prototxt.txt"
model = "MobileNet/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(weights, model)

print("-Loading video")
vs = cv2.VideoCapture(args["input"])
writer = None
(w, h) = (None, None)

print("-Running detection")
while True:
	(check, frame) = vs.read()
	if not check:
		break

	frame = imutils.resize(frame, width=500)
	if w is None or h is None:
		(h, w) = frame.shape[:2]

	# Pass the the frame through the net
	blob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)),
		0.007843,
		(300, 300),
		127.5
	)
	net.setInput(blob)
	start = time.time()
	detections = net.forward()

	# Loop through all detections in this frame
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		class_idx = int(detections[0, 0, i, 1])
		if confidence < 0.2:
			continue
		if class_idx not in traffic_idxs:
			continue
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(start_x, start_y, end_x, end_y) = box.astype("int")
		label = all_classes[class_idx]
		prob = confidence * 100
		txt = " {}: {:.2f}%".format(label, prob)
		cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[class_idx], 2)
		y = start_y - 15 if start_y - 15 > 15 else start_y + 15
		cv2.putText(frame, txt, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_idx], 2)

	# Save this frame to the output video
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(
			args["output"],
			fourcc, 30,
			(frame.shape[1],
			frame.shape[0]),
			True
		)
	writer.write(frame)

	# Display the video on desktop
	if args["playvideo"]:
		cv2.imshow("Video", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

# End while True reading frames

print("-Detection ended")

vs.release()
cv2.destroyAllWindows()


##
