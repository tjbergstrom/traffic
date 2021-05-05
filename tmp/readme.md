

- main.py
	- The main script to run the traffic detection processing.
	- Run with defaults:
		- python3 main.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi
- detect_and_track_traffic.py
	- Read video, detect and track vehicles, save output video.
- centeroid_tracker.py and trackable_object.py
	- For object tracking.
- mobilenet.py
	- MobileNet stuff.
- videostream.py
	- Multi-threaded video stream reading and other video processing.
- detectors/
	- Location of MobileNet.
- vid_inputs/
	- Location of input videos you want to process.
- vid_outputs/
	- Location to save output processed videos.


