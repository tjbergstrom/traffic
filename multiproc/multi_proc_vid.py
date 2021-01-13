# multi_proc_vid.py
# $ python3 multi_proc_vid.py
# Dec 31 2020
#
# Notes:
# This is a cool way to speed up video processing with multiprocessing.
# It's like splitting up blocks of video frames evenly among all available processors,
# and each process individually reads their block of frames, and then the blocks
# all get recombined in order into the final video.
# But it raises a problem for object tracking:
# you need to read frames contiguously for object tracking to make any sense.
# One solution would be to first process the entire video with object detection only.
# So for example, faster RCNN takes 20 seconds per frame, so multiprocess that first, and
# save the processed video with the detections drawn on the frames. Maybe the detections
# can be saved as small bright red dots. So then you can read the video again
# with lightweight object detection looking for the red dots, so you can
# process all of the video frames contiguously for object tracking.
# And it should be faster than doing tracking with RCNN detection at once linearly.


import os
import cv2
import time
import imutils
import subprocess as sp
import multiprocessing as mp
from yolo_proc import Yolo_Detection


def recombine_frames(processes):
    print("Recombining frames...")
    tmp_files = [f"tmp_{i}.mp4" for i in range(processes)]
    f = open("tmp_files.txt", "w")
    for i in tmp_files:
        f.write(f"file {i} \n")
    f.close()
    cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmp_files.txt -vcodec copy {out_vid}"
    sp.Popen(cmd, shell=True).wait()
    for i in tmp_files:
        os.remove(i)
    os.remove("tmp_files.txt")


def read_video(proc_num):
    print(f"Process: {proc_num}, start frame {jump_unit*proc_num}/{frames}")
    vs = cv2.VideoCapture(in_vid)
    vs.set(cv2.CAP_PROP_POS_FRAMES, jump_unit * proc_num)
    proc_frames = 0
    writer = cv2.VideoWriter(f"tmp_{proc_num}.mp4", fourcc, fps, (w,h), True)
    while proc_frames < jump_unit:
        check, frame = vs.read()
        if not check or frame is None:
            break
        writer.write(YD.detect(imutils.resize(frame, w)))
        proc_frames += 1
        if proc_frames % (jump_unit//2) == 0:
            print(f"Process {proc_num} 50% complete")
    vs.release()
    writer.release()
    print(f"Process {proc_num} 100% complete")


def multi_process():
    p = mp.Pool(processes)
    p.map(read_video, range(processes))
    recombine_frames(processes)


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


if __name__ == "__main__":
    start = time.time()
    width = 360
    #in_vid = "../vid_inputs/vid16.mp4"
    in_vid = "inputs/vid1.mp4"
    out_vid = "tmp.mp4"
    w, h, fps, frames = meta_info(in_vid, width)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    processes = mp.cpu_count()
    jump_unit = frames // processes
    YD = Yolo_Detection(w, h)
    multi_process()
    print(f"Finished: {time.time()-start:2f} seconds")



##
