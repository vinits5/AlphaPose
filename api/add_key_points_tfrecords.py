import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
import torch
import time
import os
import platform
import sys
import math
import cv2
import numpy as np
from api.setup import SingleImageAlphaPose, get_args
from detector.apis import get_detector
from tqdm import tqdm


def _parse(proto):
	armmapping = [0, 1, 3, 3, 2, 2]

	keys_to_features = {
		'cloth' : tf.io.FixedLenFeature([], tf.string),
		'clothMask' : tf.io.FixedLenFeature([], tf.string),
		'person' : tf.io.FixedLenFeature([], tf.string),
		'personMask' : tf.io.FixedLenFeature([], tf.string),
		'densepose' : tf.io.FixedLenFeature([], tf.string),
		'torso_gt' : tf.io.FixedLenFeature([], tf.string),
		'full' : tf.io.FixedLenFeature([], tf.string),
		'lua' : tf.io.FixedLenFeature([], tf.string),
		'rua' : tf.io.FixedLenFeature([], tf.string),
		'shoulder_left' : tf.io.FixedLenFeature([], tf.string),
		'shoulder_right' : tf.io.FixedLenFeature([], tf.string),
		'personno' : tf.io.FixedLenFeature([], tf.int64),
		'clothno' : tf.io.FixedLenFeature([], tf.int64),
		'densepose': tf.io.FixedLenFeature([], tf.string),
		"sleeves": tf.io.FixedLenFeature([], tf.string),
		'pose' : tf.io.FixedLenFeature([], tf.string),
		'heatmap' : tf.io.VarLenFeature(tf.float32),
	}

	parsed_features = tf.io.parse_single_example(proto, keys_to_features)
	cloth = tf.cast(tf.image.decode_jpeg(parsed_features['cloth'], channels=3), tf.float32)
	cloth_mask = tf.cast(tf.image.decode_jpeg(parsed_features['clothMask'], channels=1), tf.float32) 
	clotharmseg = tf.gather(armmapping, tf.cast(cloth_mask,tf.int32))
	densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
	person = tf.cast(tf.image.decode_jpeg(parsed_features['person'], channels=3), tf.float32)

	torso_gt = tf.cast(tf.image.decode_png(parsed_features['torso_gt'], channels=1), tf.float32)
	full = tf.cast(tf.image.decode_png(parsed_features['full'], channels=1), tf.float32)

	grapy = tf.cast(tf.image.decode_png(parsed_features['personMask'], channels=1), tf.float32)
	l_arm_gt = tf.cast(tf.image.decode_png(parsed_features['lua'], channels=1), tf.float32)
	r_arm_gt = tf.cast(tf.image.decode_png(parsed_features['rua'], channels=1), tf.float32)
	shoulder_left = tf.cast(tf.image.decode_png(parsed_features['shoulder_left'], channels=1), tf.float32)
	shoulder_right = tf.cast(tf.image.decode_png(parsed_features['shoulder_right'], channels=1), tf.float32)

	heatmap = tf.sparse.to_dense(parsed_features['heatmap'], default_value=0)
	heatmap = tf.reshape(heatmap, [19])

	data = {
		'cloth': cloth,
		'clothMask': cloth_mask,
		'clotharmseg': clotharmseg,
		'person': person,
		'personMask': grapy,
		'torso_gt': torso_gt,
		'full': full,
		'lua': l_arm_gt,
		'rua': r_arm_gt,
		'shoulder_left': shoulder_left,
		'shoulder_right': shoulder_right,
		'personno' : parsed_features['personno'],
		'clothno' : parsed_features['clothno'],
		'densepose': densepose,
		'sleeves': parsed_features['sleeves'],
		'pose': parsed_features['pose'],
		'heatmap': heatmap,
	}
	return data

def create_dataset(path):
	dataset = tf.data.TFRecordDataset(path)
	dataset = dataset.map(_parse, num_parallel_calls=8)
	return dataset

# Functions to create TFRecords
def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	try:
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	except:
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(cloth, clothMask, clotharmseg, person, personMask, 
					  torso_gt, full, l_arm_gt, r_arm_gt, shoulder_left, 
					  shoulder_right, key_pts, clothno, personno, 
					  densepose, sleeves, pose, heatmap):
	feature = {
		'cloth': _bytes_feature(cloth),
		'clothMask': _bytes_feature(clothMask),
		'clotharmseg': _bytes_feature(clotharmseg),
		'person': _bytes_feature(person),
		'personMask': _bytes_feature(personMask),
		'torso_gt': _bytes_feature(torso_gt),
		'full': _bytes_feature(full),
		'lua': _bytes_feature(l_arm_gt),
		'rua': _bytes_feature(r_arm_gt),
		'shoulder_left': _bytes_feature(shoulder_left),
		'shoulder_right': _bytes_feature(shoulder_right),
		'key_pts': _float_feature(key_pts),
		'clothno': _int64_feature(clothno),
		'personno': _int64_feature(personno),
		'densepose': _bytes_feature(densepose),
		'sleeves': _bytes_feature(sleeves),
		'pose': _bytes_feature(pose),
		'heatmap': _float_feature(heatmap),
	}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def evaluate(args, cfg, demo, image):
	# Estimate key-points and scores.
	im_name = 'img.png'
	pose = demo.process(im_name, image)

	if pose is None:
		key_points = np.zeros((17, 2))
		flag = False
	else:
		# Access key-points and store them.
		key_points = pose['result'][0]['keypoints'].detach().cpu().numpy()
		score = pose['result'][0]['kp_score'].detach().cpu().numpy()
		score_binary = score > 0.4
		key_points = key_points * score_binary
		flag = True
		
	key_points = key_points.astype(np.int32)
	return key_points, flag
	
def preprocess_images(data):
	image = np.array(data[0])
	return image.astype(np.uint8)

def modify_tfrecord(path, store_path):
	# Create Network.
	args, cfg = get_args()
	demo = SingleImageAlphaPose(args, cfg)

	dataset = create_dataset(path)
	BATCH_SIZE = 1
	SHUFFLE_BUFFER_SIZE = 60
	val_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
	val_iterator = iter(val_dataset)

	person_images_data = []
	dataset_size = 0

	with tf.io.TFRecordWriter(store_path) as writer:
		for idx, data in enumerate(tqdm(val_iterator)):
			if idx == 100: break
			print("Running Index: {}\r".format(idx), end="")

			cloth = preprocess_images(data['cloth'])
			clothMask = preprocess_images(data['clothMask'])
			clotharmseg = preprocess_images(data['clotharmseg'])
			person = preprocess_images(data['person'])
			personMask = preprocess_images(data['personMask'])
			torso_gt = preprocess_images(data['torso_gt'])
			full = preprocess_images(data['full'])
			l_arm_gt = preprocess_images(data['lua'])
			r_arm_gt = preprocess_images(data['rua'])
			shoulder_left = preprocess_images(data['shoulder_left'])
			shoulder_right = preprocess_images(data['shoulder_right'])
			densepose = preprocess_images(data['densepose'])

			cloth = tf.io.encode_jpeg(cloth, format='rgb')
			clothMask = tf.io.encode_png(clothMask)
			clotharmseg = tf.io.encode_png(clotharmseg)
			personMask = tf.io.encode_png(personMask)
			torso_gt = tf.io.encode_png(torso_gt)
			full = tf.io.encode_png(full)
			l_arm_gt = tf.io.encode_png(l_arm_gt)
			r_arm_gt = tf.io.encode_png(r_arm_gt)
			shoulder_left = tf.io.encode_png(shoulder_left)
			shoulder_right = tf.io.encode_png(shoulder_right)
			densepose = tf.io.encode_png(densepose)

			clothno = int(data['clothno'].numpy())
			personno = int(data['personno'].numpy())
			sleeves = data['sleeves'].numpy()[0]
			pose = data['pose'].numpy()[0].decode('UTF-8')
			heatmap = data['heatmap']

			if pose == '-' or pose == 'back':
				continue
			else:
				pose = bytes(pose, 'UTF-8')
				key_pts, flag = evaluate(args, cfg, demo, data['person'])
				if flag:
					person = tf.io.encode_jpeg(person, format='rgb')

					example = serialize_example(cloth, clothMask, clotharmseg, person, personMask, 
												torso_gt, full, l_arm_gt, r_arm_gt, shoulder_left,
												shoulder_right, key_pts, clothno, personno,
												densepose, sleeves, data['pose'], heatmap)
					writer.write(example)
					dataset_size += 1

	print("Size of the dataset: ",dataset_size)     # 21070 / 20970

if __name__ == '__main__':
	path = './uplara_tops_v11_pose_data.record'
    store_path = 'uplara_tops_v11_pose_data_key_pts.record'
    images = modify_tfrecord(path, store_path)
