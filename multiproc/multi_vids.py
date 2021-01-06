# multi_vids.py
# $ python3 multi_vids.py
# Jan 2021
#
# Notes:
# It's a different solution to multi-processing that will work for object tracking.
# Just process all of the videos at the same time,
# rather than trying to multi-process one individual video.


import os
import cv2
import time
import imutils
import subprocess as sp
import multiprocessing as mp
from yolo_proc import Yolo_Detection


def multi_process(processes):
    p = mp.Pool(processes)
    p.map(read_video, range(processes))


def meta_info(vid, width=None):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width is None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        (h, w) = imutils.resize(cap.read()[1], width=width).shape[:2]
    cap.release()
    return w, h, fps, frames, cv2.VideoWriter_fourcc("m", "p", "4", "v")


def read_video(proc_num, width=180):
    in_vid = f"inputs/vid{proc_num}.mp4"
    if not os.path.isfile(in_vid):
        return
    print(f"Processing: {in_vid}")
    w, h, fps, frames, fourcc = meta_info(in_vid, width)
    YD = Yolo_Detection(w, h)
    vs = cv2.VideoCapture(in_vid)
    writer = cv2.VideoWriter(f"vid{proc_num}_out.mp4", fourcc, fps, (w, h), True)
    proc_frames = 0
    while True:
        check, frame = vs.read()
        if not check or frame is None:
            break
        writer.write(YD.detect(imutils.resize(frame, width)))
        proc_frames += 1
        if proc_frames == (frames//2):
            print(f"Processing: {in_vid} 50% complete")
    vs.release()
    #writer.release()
    print(f"Processing: {in_vid} 100% complete")


if __name__ == "__main__":
    start = time.time()
    multi_process(mp.cpu_count())
    print(f"{time.time() - start:5f} seconds")



##
