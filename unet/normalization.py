import numpy as np
import pandas as pd
import cv2
from skimage import io

# Resized input for certain UNet that doesn't support Cropping2D
resize_height = 512
resize_width = 512

# original source img size
src_height = 512
src_width = 512

def normalize_img(filepath, resize=False):
	img = cv2.imread(filepath)
	img = img.astype(np.float32)
	subtract_mean(img)
	if resize:
		img = _resize(img)
	return img

def subtract_mean(img):
	""" assumes channel last ordering """
	img[:,:,0] -= 103.939
	img[:,:,1] -= 116.779
	img[:,:,2] -= 123.68

# TODO which interpolation mode is best?
def _resize(img):
	height, width = img.shape[:2]
	if height == resize_height and width == resize_width:
		return img
	""" resize to input dimensions required by UNet """
	img = cv2.resize(img, (resize_width, resize_height), interpolation = cv2.INTER_CUBIC)
	return img

def upsize_to_original(img):
	""" mask_img: output image from UNet is of a smaller size than the source """
	img = cv2.resize(img, (src_width, src_height), interpolation = cv2.INTER_CUBIC)
	return img
