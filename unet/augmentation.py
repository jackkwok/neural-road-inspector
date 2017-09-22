import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import random

def apply_augment_sequence(image_set_x, image_set_y):
	"""
		Randomly flip and rotate the images in both set with deterministic order.  This turns 1 image into 8 images.

		Parameters:
			image_set_x: List of Images (X) to augment
			image_set_y: List of corresponding Y image to augment in the same deterministic order applied to image_set_x

		Returns:
			image_setx_aug, image_sety_aug : augmented versions of the inputs
	"""

	# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
	# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq = iaa.Sequential(
		[
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			sometimes(iaa.Affine(
				rotate=(90, 90),
			))
		],
		random_order=False)
	seq_det = seq.to_deterministic()
	image_setx_aug = seq_det.augment_images(image_set_x)
	image_sety_aug = seq_det.augment_images(image_set_y)
	return image_setx_aug, image_sety_aug

def random_gaussian_blur(image_x, radius=5, max_sigma=3, probability=0.5):
	"""
		Randomly blur an image.

		Parameters:
			image_x: single target image
			radius: blur radius
			max_sigma: blur level randomly chosen between 1 and max_sigma, inclusive
			probability: probability where blur is applied

		Returns:
			Gassian blurred version of image with certain probability.  
			Returns original image the rest of the time.
	"""
	if random.random() < probability:
		return cv2.GaussianBlur(image_x, (radius, radius), random.randint(1, max_sigma + 1))
	else:
		return image_x

