





from mobilenet import Mnet
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def read_video(input_vid, output_vid, playvid):
	"""
	Read input video frames, draw predictions on output video frames
	Save the output frames to a video and/or display frames to desktop

	"""

	mobile_net = Mnet()
	net = mobile_net.net
	colors = mobile_net.colors

	vs = cv2.VideoCapture(input_vid)
	writer = None
	(w, h) = (None, None)

	# Read each frame in the video
	while True:
		(check, frame) = vs.read()
		if not check:
			break
		frame = imutils.resize(frame, width=720)
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
		detections = net.forward()

		# Loop through all detections in this frame and draw on the frame
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			class_idx = int(detections[0, 0, i, 1])
			if confidence < 0.2:
				continue
			if class_idx not in mobile_net.traffic_idxs:
				continue
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_x, start_y, end_x, end_y) = box.astype("int")
			label = mobile_net.all_classes[class_idx]
			prob = confidence * 100
			txt = f"{label}: {prob:2f}%"
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[class_idx], 2)
			y = start_y - 15 if start_y - 15 > 15 else start_y + 15
			cv2.putText(frame, txt, (start_x, y), 0, 0.5, colors[class_idx], 2)

		# Save this frame to the output video
		if output_vid:
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(
					output_vid,
					fourcc, 30,
					(frame.shape[1],
					frame.shape[0]),
					True
				)
			writer.write(frame)

		# Display the video on desktop
		if playvid:
			cv2.imshow("Video", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	# End while true reading frames
# End read_video()



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=False)
	ap.add_argument("-v", "--playvid", type=bool, default=False)
	args = vars(ap.parse_args())

	read_video(args["input"], args["output"], args["playvid"])



##
