

import os
import sys
import cv2


indir = "vid_data"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

for i, vid_file in enumerate(sorted(os.listdir(indir))):
	print(f"Saving: {vid_file}")
	writer = None
	outvid = f"{indir}/{i}.mp4"
	vs = cv2.VideoCapture(f"{indir}/{vid_file}")
	w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = vs.get(cv2.CAP_PROP_FPS)
	while True:
		ret, frame = vs.read()
		if frame is None:
			break
		if writer is None:
			writer = cv2.VideoWriter(outvid, fourcc, fps, (w, h), True)
		if writer:
			writer.write(frame)
		else:
			sys.exit(1)
	vs.release()
	writer.release()



##
