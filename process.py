

import os
import cv2


indir = "vid_data"
outvid = "vid_inputs/lidarvid.mp4"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None

for vid_file in sorted(os.listdir(indir)):
	print(f"Saving: {vid_file}")
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
		writer.write(frame)
	vs.release()



##
