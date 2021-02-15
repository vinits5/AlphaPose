import argparse
import torch
import os
import platform
import sys
import math
import cv2
import numpy as np
from api.setup import SingleImageAlphaPose, get_args
from detector.apis import get_detector
from tqdm import tqdm


def key_sort(element):
	return int(element.split('.')[0])

def evaluate(image_files_path, outputpath, save_img, save_keypts, save_json):	
	if not os.path.exists(outputpath + '/vis'):
		os.makedirs(outputpath + '/vis')

	# Create Network.
	args, cfg = get_args()
	demo = SingleImageAlphaPose(args, cfg)
	image_files = [os.path.join(image_files_path, x) for x in sorted(os.listdir(image_files_path), key=key_sort)]
	# image_files = ['../tf-pose-estimation/uplara_tops_data/421.jpg']

	# import ipdb; ipdb.set_trace()
	faulty_images = []
	count = 0

	for im_name in tqdm(image_files):
		# Give path of the image.
		image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)

		# Estimate key-points and scores.
		pose = demo.process(im_name, image)

		if pose is None:
			key_points = np.zeros((17, 2))
			faulty_images.append(count)
		else:
			# Access key-points and store them.
			key_points = pose['result'][0]['keypoints'].detach().cpu().numpy()
			score = pose['result'][0]['kp_score'].detach().cpu().numpy()
			score_binary = score > 0.4
			key_points = key_points * score_binary
		key_points = key_points.astype(np.int32)

		if save_keypts: np.savez(os.path.join(outputpath, 'vis', os.path.basename(im_name).split('.')[0]+'.npz'), key_points = key_points)

		if save_img: 
			img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
			img = demo.vis(img, pose)   # visulize the pose result
			cv2.imwrite(os.path.join(outputpath, 'vis', os.path.basename(im_name)), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		
		# if you want to vis the img:
		# import matplotlib.pyplot as plt
		# plt.imshow(img)
		# plt.show()

		# write the result to json:
		if save_json:
			result = [pose]
			demo.writeJson(result, outputpath, form=args.format, for_eval=args.eval)
		count += 1

	np.savez(os.path.join(outputpath, 'faulty_image_idx.npz'), data=np.array(faulty_images))

if __name__ == "__main__":
	image_files_path = '../tf-pose-estimation/uplara_tops_data'
	outputpath = 'examples/res/'
	save_img = False
	save_keypts = True
	save_json = False
	evaluate(image_files_path, outputpath, save_img, save_keypts, save_json)
