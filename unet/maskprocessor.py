import cv2
import numpy as np

def get_mask(img, target):
	"""get a binary mask [0, 1] filtered by target color."""
	tolerenace = 10
	mask = cv2.inRange(img, target-tolerenace, target+tolerenace)
	return (mask != 0)

def get_street_mask(img):
	"""img assumed to be in BGR order. street colors assumed to MapBox color scheme"""
	# Orange
	target1 = np.array([99, 160, 255])
	# Yellow
	target2 = np.array([100, 209, 242])
	# White
	target3 = np.array([255, 255, 255])

	mask1 = get_mask(img, target1)
	mask2 = get_mask(img, target2)
	mask3 = get_mask(img, target3)

	#print(np.unique(mask1))
	#print(np.unique(mask2))
	#print(np.unique(mask3))

	super_mask = np.logical_or(np.logical_or(mask1, mask2), mask3)
	return super_mask
