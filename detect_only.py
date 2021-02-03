# detect_only.py
# Jan 2021
#
# $ python3 detect_only.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi
#
# This is a multi-processed solution for processing a video with detection only.
# You split the video frames into blocks of frames among all available processors.
# So each block gets processed simultaneously.
# And then recombine the blocks of frames into the original order.
# Using the Yolo detector, but any could be used, just need to input/output a frame here


import os
import sys
import cv2
import time
import imutils
import argparse
import subprocess as sp
import multiprocessing as mp
from traffyc.yolo_proc import Yolo_Detection


def recombine_frames():
    verbose("Recombining frames...")
    tmp_files = [f"tmp_{i}.mp4" for i in range(processes)]
    f = open("tmps.txt", "w")
    for i in tmp_files:
        f.write(f"file {i} \n")
    f.close()
    cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmps.txt -vcodec copy {out_vid}"
    sp.Popen(cmd, shell=True).wait()
    for i in tmp_files:
        os.remove(i)
    os.remove("tmps.txt")


def read_video(proc_num):
    verbose(f"Process: {proc_num}, start frame {jump_unit*proc_num}/{frames}")
    vs = cv2.VideoCapture(in_vid)
    vs.set(cv2.CAP_PROP_POS_FRAMES, jump_unit * proc_num)
    proc_frames = 0
    writer = cv2.VideoWriter(f"tmp_{proc_num}.mp4", fourcc, fps, (w, h), True)
    while proc_frames < jump_unit:
        check, frame = vs.read()
        if not check or frame is None:
            break
        writer.write(YD.detect(imutils.resize(frame, w)))
        proc_frames += 1
        if proc_frames == (jump_unit // 2):
            verbose(f"Process {proc_num} 50% complete")
    vs.release()
    writer.release()
    verbose(f"Process {proc_num} 100% complete")


def multi_process():
    p = mp.Pool(processes)
    p.map(read_video, range(processes))
    recombine_frames()


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
    return w, h, fps, frames


def verbose(msg):
    if v:
        print(msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=False)
    ap.add_argument("-w", "--width", type=int, default=None)
    ap.add_argument("-v", "--verbose", type=bool, default=True)
    args = vars(ap.parse_args())
    in_vid = args["input"]
    out_vid = args["output"]
    v = args["verbose"]
    w, h, fps, frames = meta_info(in_vid, args["width"])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    processes = min(mp.cpu_count(), frames)
    if processes == 0:
        sys.exit(f"No processors found")
    jump_unit = frames // processes
    YD = Yolo_Detection(w, h)
    multi_process()



##
