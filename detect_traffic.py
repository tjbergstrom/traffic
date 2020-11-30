# detect_traffic.py
# November 2020
# Input a video, and use MobileNet to detect:
#	person, car, bus, bicycle, motorcycle
# The detections (lables and %s) are drawn on output frames
# With bounding boxes around the detected object
# Options to display the video on desktop and/or save the ouput video
#
# Run:
# Just save the output video:
# python3 detect_traffic.py -i vid_inputs/vid3.mp4 -o vid_outputs/0.avi
# Just watch the output video:
# python3 detect_traffic.py -i vid_inputs/vid3.mp4 -v tru
#  (hit 'q' to quit the video)


from videostream import Video_Thread
from mobilenet import Mnet
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
import os


def read_video(input_vid, output_vid, playvid):
	"""
	Read input video frames, draw predictions on output video frames
	Save the output frames to a video and/or display frames to desktop

	"""

	mobile_net = Mnet()
	net = mobile_net.net
	colors = mobile_net.colors

	vs = Video_Thread(input_vid).start()
	fps = vs.fps()
	writer = None
	(w, h) = (None, None)

	# Read each frame in the video
	print("Reading video...")
	while True:
		frame = vs.read()
		if frame is None:
			if not vs.more_frames():
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

		# Loop through all detections in this frame and draw them on the frame
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
			txt = f"{label}: {prob:.2f}%"
			cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), colors[class_idx], 2)
			y = start_y - 16 if start_y - 16 > 16 else start_y + 16
			cv2.putText(frame, txt, (start_x,y), 0, 0.5, colors[class_idx], 2)

		# Save this frame to the output video
		if output_vid:
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(output_vid, fourcc, fps, (w, h), True)
			writer.write(frame)

		# Display the video on desktop
		if playvid:
			cv2.imshow("Video", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				vs.release()
				vs.stop()
				break

	# End while true reading frames
# End read_video()



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=False)
	ap.add_argument("-v", "--playvid", type=bool, default=False)
	args = vars(ap.parse_args())

	if not os.path.isfile(args["input"]):
		print("\'{}\' is not a filepath".format(args["input"]))
		sys.exit(1)
	if args["output"] and not os.path.isdir(os.path.dirname(args["output"])):
		print("Cannot save an output video to \'{}\'".format(args["output"]))
		sys.exit(1)
	if args["output"] and os.path.isfile(args["output"]):
		print("Warning: will be over-writing output video \'{}\'".format(args["output"]))
		time.sleep(3.0)

	read_video(args["input"], args["output"], args["playvid"])

	print("Finished reading video")
	if args["output"]:
		if os.path.isfile(args["output"]):
			print("Output video successfully saved")
		else:
			print("Output video not saved")
			sys.exit(1)
	print("Task failed successfully")



##
