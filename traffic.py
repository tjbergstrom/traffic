# traffic.py
# December 2020
#
# Run:
# Detection and tracking:
# Just save the processed output video:
# $ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi pt tru
# Just watch the output video:
# $ python3 traffic.py -i vid_inputs/vid0.mp4 -p tru -t tru
#   (hit 'q' to quit the video)
# Detection only, save the processed output video:
# $ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi -d tru


from detect_and_track_traffic import Traffic_Detection
import argparse
import time
import sys
import os


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="input video filepath")
	ap.add_argument("-o", "--output", required=False, help="save output video filepath")
	ap.add_argument("-p", "--playvid", type=bool, default=False, help="play video while processing")
	ap.add_argument("-w", "--width", type=int, default=None, help="resize frame width")
	ap.add_argument("-f", "--freq", type=int, default=5, help="detection frequency, every _ frames")
	ap.add_argument("-d", "--detect_only", type=bool, default=False, help="no tracking")
	ap.add_argument("-t", "--detect_and_track", type=bool, default=False, help="increased performance")
	args = vars(ap.parse_args())

	in_vid = args["input"]
	out_vid = args["output"]
	width = args["width"]
	freq = args["freq"]

	if not os.path.isfile(in_vid):
		sys.exit(f"\'{in_vid}\' is not a filepath")
	if out_vid and not os.path.isdir(os.path.dirname(out_vid)):
		sys.exit(f"Cannot save an output video to \'{out_vid}\'")
	if out_vid and not os.path.basename(out_vid):
		sys.exit(f"No output file specified \'{out_vid}\'")
	if not out_vid and not args["playvid"]:
		sys.exit(f"Not saving or displaying output?")
	if width and (width < 360 or width > 1800):
		sys.exit(f"Width \'{width}\' out of range")
	if freq < 2 or freq > 40:
		sys.exit(f"Detection frequency \'{freq}\' not supported")
	if out_vid and os.path.isfile(out_vid):
		print(f"Warning: will be over-writing output video \'{out_vid}\'")
		time.sleep(3.0)

	start = time.time()
	if args["detect_and_track"]:
		Traffic_Detection(width, freq).read_video(in_vid, out_vid, args["playvid"])
	elif args["detect_only"]:
		if width:
			os.system(f"python3 detect_only.py -i {in_vid} -o {out_vid} -w {width}")
		else:
			os.system(f"python3 detect_only.py -i {in_vid} -o {out_vid}")
	else:
		sys.exit(f"Need an arg -d or -t")
	print(f"Finished processing video ({time.time()-start:.2f} seconds)")

	if out_vid:
		if os.path.isfile(out_vid):
			print(f"Output video successfully saved")
		else:
			sys.exit(f"Output video not saved")
	print(f"Finished successfully")



##
