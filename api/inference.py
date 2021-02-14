import argparse
import torch
import os
import platform
import sys
import math
import cv2
import numpy as np
from api.setup import SingleImageAlphaPose, get_args


def evaluate(save_img, save_keypts, save_json):
	outputpath = "examples/res/"
	if not os.path.exists(outputpath + '/vis'):
		os.mkdir(outputpath + '/vis')

	# Create Network.
	args, cfg = get_args()
	demo = SingleImageAlphaPose(args, cfg)

	# Give path of the image.
	im_name = 'examples/demo/80.png'
	image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)

	# Estimate key-points and scores.
	pose = demo.process(im_name, image)

	# Access key-points and store them.
	key_points = pose['result'][0]['keypoints'].detach().cpu().numpy()
	score = pose['result'][0]['kp_score'].detach().cpu().numpy()
	score_binary = score > 0.4
	key_points = key_points * score_binary
	key_points = key_points.astype(np.int32)

	if save_keypts: np.savez(os.path.join(outputpath, os.path.basename(im_name).split('.')[0]+'.npz'), key_points = key_points)

	img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
	img = demo.vis(img, pose)   # visulize the pose result
	if save_img: cv2.imwrite(os.path.join(outputpath, 'vis', os.path.basename(im_name)), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	
	# if you want to vis the img:
	# import matplotlib.pyplot as plt
	# plt.imshow(img)
	# plt.show()

	# write the result to json:
	# if save_json:
	# 	result = [pose]
	# 	demo.writeJson(result, outputpath, form=args.format, for_eval=args.eval)

if __name__ == "__main__":
	save_img = True
	save_keypts = True
	save_json = False
	evaluate(save_img, save_keypts, save_json)
