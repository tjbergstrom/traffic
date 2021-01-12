# traffic.py
# December 2020
#
# Run:
# Just save the output video:
# $ python3 traffic.py -i vid_inputs/vid0.mp4 -o vid_outputs/0.avi
# Just watch the output video:
# $ python3 traffic.py -i vid_inputs/vid0.mp4 -v tru
#   (hit 'q' to quit the video)


from detect_and_track_traffic import Traffic_Detection
import argparse
import time
import sys
import os


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True)
	ap.add_argument("-o", "--output", required=False)
	ap.add_argument("-v", "--playvid", type=bool, default=False)
	ap.add_argument("-w", "--width", type=int, default=720)
	args = vars(ap.parse_args())

	in_vid = args["input"]
	out_vid = args["output"]

	if not os.path.isfile(in_vid):
		sys.exit(f"\'{in_vid}\' is not a filepath")
	if out_vid and not os.path.isdir(os.path.dirname(out_vid)):
		sys.exit(f"Cannot save an output video to \'{out_vid}\'")
	if out_vid and not os.path.basename(out_vid):
		sys.exit(f"No output file specified \'{out_vid}\'")
	if not out_vid and not args["playvid"]:
		sys.exit(f"Not saving or displaying output?")
	if out_vid and os.path.isfile(out_vid):
		print(f"Warning: will be over-writing output video \'{out_vid}\'")
		time.sleep(3.0)

	start = time.time()
	Traffic_Detection(args["width"]).read_video(in_vid, out_vid, args["playvid"])

	print(f"Finished processing video ({time.time()-start:.2f} seconds)")
	if out_vid:
		if os.path.isfile(out_vid):
			print("Output video successfully saved")
		else:
			sys.exit("Output video not saved")
	print("Task failed successfully")



##
