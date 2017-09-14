import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def apply_augment_sequence(image_set_x, image_set_y):
	"""Randomly flip the images in both set with deterministic order in both sets"""
	seq = iaa.Sequential(
		[
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
		],
		random_order=False)
	seq_det = seq.to_deterministic()
	image_setx_aug = seq_det.augment_images(image_set_x)
	image_sety_aug = seq_det.augment_images(image_set_y)
	return image_setx_aug, image_sety_aug
