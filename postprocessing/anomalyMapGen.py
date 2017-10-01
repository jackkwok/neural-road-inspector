import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

def _realign_post_mask(img):
	# WARNING: warp_matrix is calibrated for digitalglobe post-harvey images.
	# (digitalglobe post-harvey) needs to be shifted about 6 pixels to the left and 6 pixels to the bottom to match img1.
	warp_matrix = np.float32([[1,0,6],[0,1,-6]])
	rows,cols,channels = img.shape
	return cv2.warpAffine(img, warp_matrix, (cols,rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

def _get_diff(pre_mask_path, post_mask_path):
	pre_mask = cv2.imread(pre_mask_path)
	# taking care of antialiasing/interpolations in the mask image
	pre_mask[pre_mask > 126] = 255
	pre_mask[pre_mask <= 126] = 0

	post_mask = cv2.imread(post_mask_path)
	post_mask[post_mask > 126] = 255
	post_mask[post_mask <= 126] = 0

	# Due to misalignment between pre and post images, we run ECC alignment to align the images.
	# Ran ECC algo on source satellite images and we will hardcode the realignment for now.
	# TODO: remove hardcoding and use ecc_align in alignment module.
	post_mask = _realign_post_mask(post_mask)

	anomaly_mask = pre_mask - post_mask
	anomaly_mask[anomaly_mask != 255] = 0 # takes care of negative values
	return anomaly_mask

def _add_alpha_channel_mask(img, alpha=0.80):
	"""
	Parameters:
		img: the source image with bgr channels which has black or white colored pixels only.
		alpha: the alpha transparency [0.0, 1.0]

	Returns:
		An image with BGRA channels, in that order.
	"""
	b_channel, g_channel, r_channel = cv2.split(img)
	alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
	alpha_channel[r_channel > 126] = int(255 * alpha)
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) 

def _colorize_mask(bgra_img):
	"""
	Parameter:
		bgra_img: 4 channel images which has black or white colored pixels only.
	Returns:
		colorized image where all pixels which are not black color: (0,0,0) are turned to red color
	"""
	b_channel, g_channel, r_channel, alpha_channel = cv2.split(bgra_img)
	
	b_channel[:] = 0
	g_channel[:] = 0
	r_channel[r_channel > 126] = 255
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def _filter_by_street_map(img, street_map_path):
	orig_street_img = cv2.imread(street_map_path)

	gray_street_img = cv2.cvtColor(orig_street_img, cv2.COLOR_BGR2GRAY)
	ret, binary_street_mask = cv2.threshold(gray_street_img, 250, 255, cv2.THRESH_BINARY)

	img = img.astype(int)
	return cv2.bitwise_and(img, img, mask=binary_street_mask)

# TODO: Thresholding?
# Strutural Similarly Measure: http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def generate_anomaly_img(pre_path, post_path, street_map_path):
	"""
	Parameters:
		pre_path: file path to the pre event segmentation image
		post_path: file path to the post event segmentation image
		street_map_path: file path to the street map image

	Returns:
		anomaly annotation image tile to be used in a map overlay.
	"""
	diff_img = _get_diff(pre_path, post_path)
	return _colorize_mask(_add_alpha_channel_mask(_filter_by_street_map(diff_img, street_map_path)))

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

def generateAnomalyMapTiles(pre_mask_dir, post_mask_dir, street_map_dir, output_dir, street_map_file_ext = '.png'):
	"""
		Parameters:
			pre_mask_dir: directory containing the pre event segmentation mask tiles.
			post_mask_dir: directory containing the post event segmentation mask tiles.
			street_map_dir: directory containing steet map tiles.
			output_dir: directory where new anomaly tiles are saved.
			street_map_file_ext: file extension of the street map tiles. default is .png.
	"""
	pre_file_list = _image_file_list(pre_mask_dir)
	for pre_file in tqdm(pre_file_list):
		post_file = pre_file.replace(pre_mask_dir, post_mask_dir, 1)
		if os.path.exists(post_file):
			street_file = pre_file.replace(pre_mask_dir, street_map_dir, 1)
			street_file = os.path.splitext(street_file)[0] + street_map_file_ext
			anomaly_img = generate_anomaly_img(pre_file, post_file, street_file)
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
if len(sys.argv) > 4:
	pre_mask_dir = sys.argv[1]
	post_mask_dir = sys.argv[2]
	street_map_dir = sys.argv[3]
	output_dir = sys.argv[4]

	if not os.path.isdir(pre_mask_dir):
		print ('error: invalid directory {}'.format(pre_mask_dir))
		sys.exit(0)
	if not os.path.isdir(post_mask_dir):
		print ('error: invalid directory {}'.format(post_mask_dir))
		sys.exit(0)
	if not os.path.isdir(street_map_dir):
		print ('error: invalid directory {}'.format(street_map_dir))
		sys.exit(0)

	# add OS-indepedent slashes
	pre_mask_dir = os.path.join(pre_mask_dir, '') 
	post_mask_dir = os.path.join(post_mask_dir, '')
	street_map_dir = os.path.join(street_map_dir, '')
	output_dir = os.path.join(output_dir, '')
	makedirs(output_dir)
	generateAnomalyMapTiles(pre_mask_dir, post_mask_dir, street_map_dir, output_dir)
elif len(sys.argv) == 2 and sys.argv[1] == '-h':
	print('******************** \n\n Usage: python anomalyMapGen.py <pre_event_segment_dir> <post_event_segment_dir> <street_map_dir> <output_dir>\n\n ) \n\n********************')
else:
	print ('error: required command line argument missing. \n\n Syntax: python anomalyMapGen.py <pre_event_segment_dir> <post_event_segment_dir> <street_map_dir> <output_dir>')
	sys.exit(0)

