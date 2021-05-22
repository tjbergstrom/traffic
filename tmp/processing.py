# processing.py
# May 2021
#
# python3 processing.py -i vid_data -o vid_outputs/vid.avi
#
# We received a sample video split into seven smaller videos.
# This will recombine them into the original full length video.
# But also why not just recombine them, and then run the traffic detection,
# And then compress the final video, all at once...


import os
import sys
import cviz
import argparse
import subprocess as sp


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--indir", required=True, help="input videos directory")
	ap.add_argument("-o", "--outvid", required=True, help="recombined video filepath")
	args = vars(ap.parse_args())

	# indir should be just the directory where all the .mjpegs are
	indir = args["indir"]
	# outvid should be the path/filename.avi you want to save as
	outvid = args["outvid"]

	# Recombine all the .mjpegs into one temporary .mp4 (O(1) time, no processing)
	tmpvid, ext = os.path.splitext(outvid)
	tmpvid = f"{tmpvid}.mp4"

	vidz = sorted(os.listdir(indir))
	with open("tmps.txt", "w") as f:
		for vid_path in vidz:
			vid_path = os.path.join(indir, vid_path)
			f.write(f"file {vid_path} \n")
	cmd = f"ffmpeg -y -loglevel error -f concat -safe 0 -i tmps.txt -vcodec copy {tmpvid}"
	sp.Popen(cmd, shell=True).wait()
	os.remove("tmps.txt")

	'''
	# Convert the whole thing to one .avi
	#outvid = cviz.avi_conv(tmpvid)
	# Remove the temporary .mp4 used for recombining
	#os.remove(tmpvid)
	# Stop here if you only wanted to convert the original videos to one .avi
	#sys.exit(0)
	'''

	# Run the traffic detection processing, which will save as .avi
	cmd = f"python3 detectraffic_mp.py -i {tmpvid} -o {outvid} -f 4"
	os.system(cmd)

	# Remove the temporary .mp4 used for recombining
	os.remove(tmpvid)

	vidpath, ext = os.path.splitext(outvid)

	print("Compressing... ")

	# Compress the finished processed .avi video into a much smaller .mp4
	cmd = f"ffmpeg -loglevel error -i {outvid} -vcodec h264 -acodec aac {vidpath}.mp4"
	os.system(cmd)



##
