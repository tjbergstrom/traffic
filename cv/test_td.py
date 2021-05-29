# test_td.py
# May 2021
# Test that the traffic detection processing is all good.
#
# Process and save a sample video, should only take 10 seconds.
# This is a quick way to make sure if you edited anything it still works.
#
# $ python3 -m pytest


import os
import cviz
import pytest


def detectraffic_mp():
	return "python3 detectraffic_mp.py "


def invid():
	return "vid_inputs/vid0.mp4"


def outvid():
	return "vid_outputs/testvid.avi"


def test_precheck():
	assert os.path.isfile(invid())
	mndir = "detectors/MobileNet"
	weights = f"{mndir}/MobileNetSSD_deploy.prototxt.txt"
	model = f"{mndir}/MobileNetSSD_deploy.caffemodel"
	assert os.path.isfile(weights)
	assert os.path.isfile(model)


def test_bad_args():
	if os.path.isfile(outvid()):
		os.remove(outvid())
	cmd = detectraffic_mp()
	os.system(cmd)
	assert not os.path.isfile(outvid())


def test_output_vid():
	cmd = detectraffic_mp()
	cmd += f" -i {invid()} -o {outvid()} -w 180 -f 30 "
	os.system(cmd)
	assert os.path.isfile(outvid())


def test_processed():
	assert cviz.frame_cnt(invid()) == cviz.frame_cnt(outvid())
	assert cviz.vid_fps(invid()) == cviz.vid_fps(outvid())
	os.remove(outvid())



##
