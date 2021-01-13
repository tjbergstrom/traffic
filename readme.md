

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


<br>
<br>

![alt text](https://raw.githubusercontent.com/tjbergstrom/traffic/master/vid_outputs/9.gif)






