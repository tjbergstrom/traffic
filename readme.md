


- ```detectrack.py```
    - The main script for multi-processed detection with tracking. Run it with:
        - $ python3 detectrack.py -i vid_inputs/vid9.mp4 -o vid_outputs/tmp.avi -w 500 -f 10

- ```traffic.py```
    - The main script for detection with either mp or tracking. Run it with:
        - ```$ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -d tru```
            - detection only, save the output video
        - ```$ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -t tru```
            - detection with tracking, save the output video
        - ```$ python3 traffic.py -i vid_inputs/vid0.mp4 -t tru -v tru```
            - detection with tracking, just watch the output without saving.

- ```detect_and_track_traffic.py```
    - The primary methods for processing input video frames for detecting and tracking, and preparing output video, with MobileNet detector

- ```detect_only.py```
    - Multi-Processed detection with the Yolo detector.

- ```traffyc/```
    - Helpers, including:
        - ```centeroid_tracker.py``` and ```trackable_object.py```
            - For object tracking.
        - ```videostream.py```
            - Multi-threaded solution for reading video frames.
        - ```mobilenet.py```
            - Just the needed properties of MobileNet.
        - ```yolo_proc.py```
            - Detection method, input and output a frame for the mutli-processed detection only.

- ```vid_inputs/```
    - Video files that you want to process. Should be .mp4

- ```vid_outputs/```
    - The saved processed videos, can be either .avi or .mp4


- ```detectors/```
    - The models used for detection.




<br>
<br>

![alt text](https://raw.githubusercontent.com/tjbergstrom/traffic/master/vid_outputs/9.gif)

*Detection and tracking with MobileNet!*






