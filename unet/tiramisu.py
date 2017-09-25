from __future__ import absolute_import
import os

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers import add
from keras.layers import Conv2D, Conv2DTranspose
from keras import backend as K
from keras.regularizers import l2

import numpy as np

K.set_image_dim_ordering('tf') # Tensorflow dimension ordering

class Tiramisu(object):
	def __init__(self, num_channels = 3, img_rows = 256, img_cols = 256):
		"""
		The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
		https://arxiv.org/abs/1611.09326

		Parameters:
			num_channels: the total number of channels for the data (e.g. for images, it would be 3 for RGB and 4 for RGBA)
			img_rows: number of rows for the image (height)
			img_cols: number of columns for the image (width)

		Limitation:
			img_rows and img_cols must be multples of 16!
		"""
		self.num_channels = num_channels
		self.img_rows = img_rows
		self.img_cols = img_cols

	def DenseBlock(self, layers, filters):
		model = self.model
		for i in range(layers):
			model.add(BatchNormalization(mode=0, axis=1,
										 gamma_regularizer=l2(0.0001),
										 beta_regularizer=l2(0.0001)))
			model.add(Activation('relu'))
			model.add(Conv2D(filters, 
							kernel_size=(3, 3), 
							padding='same',
							kernel_initializer="he_uniform",
							data_format='channels_last'))
			model.add(Dropout(0.2))

	def TransitionDown(self,filters):
		model = self.model
		model.add(BatchNormalization(mode=0, axis=1,
									 gamma_regularizer=l2(0.0001),
									 beta_regularizer=l2(0.0001)))
		model.add(Activation('relu'))
		model.add(Conv2D(filters, kernel_size=(1, 1), padding='same',
								  kernel_initializer="he_uniform"))
		model.add(Dropout(0.2))
		model.add(MaxPooling2D( pool_size=(2, 2),
								strides=(2, 2),
								data_format='channels_last'))

	def TransitionUp(self,filters,input_shape,output_shape):
		model = self.model
		model.add(Conv2DTranspose(filters,  kernel_size=(3, 3), strides=(2, 2),
											padding='same',
											output_shape=output_shape,
											input_shape=input_shape,
											kernel_initializer="he_uniform",
											data_format='channels_last'))

	def get_tiramisu(self):
		model = self.model = models.Sequential()
		# cropping
		# model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

		model.add(Conv2D(48, 
						kernel_size=(3, 3), 
						padding='same', 
						input_shape=(self.img_rows, self.img_cols, self.num_channels),
						kernel_initializer="he_uniform",
						kernel_regularizer = l2(0.0001),
						data_format='channels_last'))

		self.DenseBlock(5,108) # 5*12 = 60 + 48 = 108
		self.TransitionDown(108)
		self.DenseBlock(5,168) # 5*12 = 60 + 108 = 168
		self.TransitionDown(168)
		self.DenseBlock(5,228) # 5*12 = 60 + 168 = 228
		self.TransitionDown(228)
		self.DenseBlock(5,288)# 5*12 = 60 + 228 = 288
		self.TransitionDown(288)
		self.DenseBlock(5,348) # 5*12 = 60 + 288 = 348
		self.TransitionDown(348)

		self.DenseBlock(15,408) # m = 348 + 5*12 = 408

		self.TransitionUp(468, (468, self.img_rows/32, self.img_cols/32), (None, 468, self.img_rows/16, self.img_cols/16))
		self.DenseBlock(5,468)

		self.TransitionUp(408, (408, self.img_rows/16, self.img_cols/16), (None, 408, self.img_rows/8, self.img_cols/8))
		self.DenseBlock(5,408)

		self.TransitionUp(348, (348, self.img_rows/8, self.img_cols/8), (None, 348, self.img_rows/4, self.img_cols/4))
		self.DenseBlock(5,348)

		self.TransitionUp(288, (288, self.img_rows/4, self.img_cols/4), (None, 288, self.img_rows/2, self.img_cols/2))
		self.DenseBlock(5,288)

		self.TransitionUp(228, (228, self.img_rows/2, self.img_cols/2), (None, 228, self.img_rows, self.img_cols))
		self.DenseBlock(5,228)

		model.add(Conv2D(12, 
						kernel_size=(1,1), 
						padding='same',
						kernel_initializer="he_uniform",
						kernel_regularizer = l2(0.0001),
						data_format='channels_last'))
		
		model.add(Reshape((12, self.img_rows * self.img_cols)))
		model.add(Permute((2, 1)))
		model.add(Activation('sigmoid'))
		#model.summary()
		return model
