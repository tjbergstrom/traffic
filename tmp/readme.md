

- detectraffic.py
	- Original traffic detection with object tracking, process and save a video.
		- python3 detectraffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi
- detectraffic_mp.py
	- Multi-processed traffic detection with object tracking, process and save a video.
		- python3 detectraffic_mp.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi
- detectrack.py
	- Input video frames, detect and track objects across frames, draw detections on output frames.
- centeroid_tracker.py and trackable_object.py
	- For object tracking.
- cviz.py
	- Other video processing stuff.
- mobilenet.py
	- MobileNet stuff.
- videostream.py
	- Multi-threaded video stream reading.
- detectors/
	- Location of MobileNet.
- vid_inputs/
	- Location of input videos you want to process.
- vid_outputs/
	- Location to save output processed videos.


