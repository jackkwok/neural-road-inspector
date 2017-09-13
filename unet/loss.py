import numpy as np
from keras.models import *
from keras.optimizers import *

#from keras.losses import binary_crossentropy

smooth = 1.

# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
	return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def binary_crossentropy_dice_loss(y_true, y_pred):
	"""Adding on BCE is supposed to make model converge faster"""
	return binary_crossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))