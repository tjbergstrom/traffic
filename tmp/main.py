# main.py
# May 2021
#
# $ python3 main.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi


from detect_and_track_traffic import Traffic_Detection
import argparse
import cviz
import time
import sys
import os


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="input video filepath")
	ap.add_argument("-o", "--output", required=True, help="save output video filepath")
	ap.add_argument("-w", "--width", type=int, default=None, help="resize video frame width")
	ap.add_argument("-f", "--freq", type=int, default=5, help="detection frequency, default is every 5 frames")
	args = vars(ap.parse_args())

	in_vid = args["input"]
	out_vid = args["output"]
	width = args["width"]
	freq = args["freq"]

	if not os.path.isfile(in_vid):
		sys.exit(f"'{in_vid}' is not a filepath")
	if not os.path.isdir(os.path.dirname(out_vid)):
		sys.exit(f"Cannot save an output video to '{out_vid}'")
	if not os.path.basename(out_vid):
		sys.exit(f"No output file specified '{out_vid}'")
	if width and (width < 180 or width > 1800):
		sys.exit(f"Width '{width}' out of range")
	if freq < 2 or freq > 20:
		sys.exit(f"Detection frequency '{freq}' not supported")
	if not cviz.valid_vidtyp(in_vid):
		sys.exit(f"Not a valid video extension, '{in_vid}'")
	if out_vid and os.path.isfile(out_vid):
		print(f"Warning: will be over-writing output video '{out_vid}'")
		time.sleep(3.0)
		os.remove(out_vid)

	start = time.time()
	Traffic_Detection(width, freq).read_video(in_vid, out_vid)
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")

	if os.path.isfile(out_vid):
		print(f"Output video successfully saved")
	else:
		sys.exit(f"Output video not saved")



##
