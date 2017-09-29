import numpy as np
import pandas as pd
import cv2

def normalize_img(filepath):
	img = cv2.imread(filepath)
	subtract_mean(img)
	return img

def subtract_mean(img):
	""" assumes channel last ordering """
	img = img.astype(np.float32)
	img[:,:,0] -= 103.939
	img[:,:,1] -= 116.779
	img[:,:,2] -= 123.68
