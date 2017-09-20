import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

def _get_diff(pre_mask_path, post_mask_path):
	pre_mask = cv2.imread(pre_mask_path)
	# taking care of antialiasing/interpolations in the mask image
	pre_mask[pre_mask > 126] = 255
	pre_mask[pre_mask <= 126] = 0

	post_mask = cv2.imread(post_mask_path)
	post_mask[post_mask > 126] = 255
	post_mask[post_mask <= 126] = 0

	# Apply mathematically morphology to reduce false positive
	# dilate the post_path or erode the pre_path
	kernel = np.ones((3,3), np.uint8)
	pre_mask = cv2.erode(pre_mask, kernel, iterations=1)
	post_mask = cv2.dilate(post_mask, kernel, iterations=1)

	# TODO: Due to misalignment between pre and post images, we run RANSAC alignment to correct the position.
	# RANSAC: https://en.wikipedia.org/wiki/Random_sample_consensus
	# Question: run algo on source satellite images or their masks?
	anomaly_mask = pre_mask - post_mask
	anomaly_mask[anomaly_mask != 255] = 0 # takes care of negative values
	return anomaly_mask

def _add_alpha_channel_mask(img, alpha=0.80):
	"""
		params:
		img: the source image which has black or white colored pixels only.
		alpha: the alpha transparency [0.0, 1.0]
	"""
	b_channel, g_channel, r_channel = cv2.split(img)
	alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
	alpha_channel[r_channel > 126] = int(255 * alpha)
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) 

def _colorize_mask(bgra_img):
	"""
		bgra_img: 4 channel images which has black or white colored pixels only.
		colorize (red) all pixels which are not black color: (0,0,0)
	"""
	b_channel, g_channel, r_channel, alpha_channel = cv2.split(bgra_img)
	
	b_channel[:] = 0
	g_channel[:] = 0
	r_channel[r_channel > 126] = 255
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

# TODO: Thresholding?
# Strutural Similarly Measure: http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def generate_anomaly_img(pre_path, post_path):
	"""
	Parameters:
		pre_path: file path to the pre event segmentation image
		post_path: file path to the post event segmentation image

	Returns:
		Image with anomaly annotation to be used in a map overlay.
	"""
	diff_img = _get_diff(pre_path, post_path)
	return _colorize_mask(_add_alpha_channel_mask(diff_img))

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def _image_file_list(dir_path):
	""" limitation: the images files must have an image extension: webp, jpg, png, or jpeg """
	result = []
	for root, dirs, files in os.walk(dir_path):
		for file in files:
			if file.endswith('.webp') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
				result.append(os.path.join(root, file))
	return result

def generateAnomalyMapTiles(pre_mask_dir, post_mask_dir, output_dir):
	pre_file_list = _image_file_list(pre_mask_dir)
	for pre_file in tqdm(pre_file_list):
		post_file = pre_file.replace(pre_mask_dir, post_mask_dir, 1)
		if os.path.exists(post_file):
			anomaly_img = generate_anomaly_img(pre_file, post_file)
			output_file = pre_file.replace(pre_mask_dir, output_dir, 1)
			# must save in a format that supports alpha transparency (e.g. PNG or WEBP)
			output_file = os.path.splitext(output_file)[0]+'.png'
			print('output file: {}'.format(output_file))
			makedirs(os.path.dirname(output_file))
			cv2.imwrite(output_file, anomaly_img, [cv2.IMWRITE_PNG_COMPRESSION, 9]) # max compression
		else:
			print('no matching post image: {}'.format(post_file))
	print('map tiles generation complete')

# execution starts here. command line args processing.
if len(sys.argv) > 3:
	pre_mask_dir = sys.argv[1]
	post_mask_dir = sys.argv[2]
	output_dir = sys.argv[3]

	if not os.path.isdir(pre_mask_dir):
		print ('error: invalid directory {}'.format(pre_mask_dir))
		sys.exit(0)
	if not os.path.isdir(post_mask_dir):
		print ('error: invalid directory {}'.format(post_mask_dir))
		sys.exit(0)

	# add OS-indepedent slashes
	pre_mask_dir = os.path.join(pre_mask_dir, '') 
	post_mask_dir = os.path.join(post_mask_dir, '')
	output_dir = os.path.join(output_dir, '')
	makedirs(output_dir)
	generateAnomalyMapTiles(pre_mask_dir, post_mask_dir, output_dir)
else:
	print ('error: required command line argument missing. Syntax: python anomalyMapGen.py <pre_dir> <post_dir> <output_dir>')
	sys.exit(0)

