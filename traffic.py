# traffic.py
# December 2020
#
# Run:
# Just save the output video:
# $ python3 traffic.py -i vid_inputs/vid8.mp4 -o vid_outputs/0.avi
# Just watch the output video:
# $ python3 traffic.py -i vid_inputs/vid8.mp4 -v tru
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
	args = vars(ap.parse_args())

	if not os.path.isfile(args["input"]):
		print("\'{}\' is not a filepath".format(args["input"]))
		sys.exit(1)
	if args["output"] and not os.path.isdir(os.path.dirname(args["output"])):
		print("Cannot save an output video to \'{}\'".format(args["output"]))
		sys.exit(1)
	if args["output"] and not os.path.basename(args["output"]):
		print("No output file specified \'{}\'".format(args["output"]))
		sys.exit(1)
	if args["output"] and os.path.isfile(args["output"]):
		print("Warning: will be over-writing output video \'{}\'".format(args["output"]))
		time.sleep(3.0)

	Traffic_Detection().read_video(args["input"], args["output"], args["playvid"])

	print("Finished reading video")
	if args["output"]:
		if os.path.isfile(args["output"]):
			print("Output video successfully saved")
		else:
			print("Output video not saved")
			sys.exit(1)
	print("Task failed successfully")



##
