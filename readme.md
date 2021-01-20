

- ```traffic.py```
    - The main script. Run it with:
        - ```$ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi``` to save an output video, or
        - ```$ python3 traffic.py -i vid_inputs/vid0.mp4 -v tru``` to just watch the output without saving anything.

- ```detect_and_track_traffic.py```
    - The primary methods for processing input video frames for detecting and tracking, and preparing output video.

- ```traffyc/```
    - Helpers, including:
        - ```centeroid_tracker.py``` and ```trackable_object.py```
            - For object tracking.
        - ```videostream.py```
            - Multi-threaded solution for reading video frames.
        - ```mobilenet.py```
            - Just the needed properties of MobileNet.

- ```vid_inputs/```
    - Video files that you want to process. Should be .mp4

- ```vid_outputs/```
    - The saved processed videos, can be either .avi or .mp4


- ```multiproc/```
    - Trying out some solutions for multi-processing videos:
        - ```multi_proc_vid.py```
            - Take a video, split up chunks of frames among available processors and run the object detection on each, then recombine the chunks of frames into the original order for an output video - works great for detection only, but not intended for tracking.
        - ```yolo_proc.py```
            - The detection processing used with multi_proc_vid.py - any detection method or processing can be used, just needs to input and output a frame.



<br>
<br>

![alt text](https://raw.githubusercontent.com/tjbergstrom/traffic/master/vid_outputs/9.gif)

*Detection and tracking with MobileNet!*






