import numpy as np
import pandas as pd
import cv2
import os
from maskprocessor import *

class CustomImgGenerator(object):
	""" A Custom Image Generator that generate
	    training set and validation set with a 8:2 split. """
	def __init__(self, training_root):
		self.training_data_root = training_root
		self.train_dir = self.training_data_root + 'mapbox-sat/'
		self.train_mask_dir = self.training_data_root + 'mapbox-street/'
		self.complete_set_df = pd.read_csv(self.train_dir + 'tile_log.csv')
		# shuffle everything in the training set. must reset the dataframe index.
		self.complete_set_df = self.complete_set_df.sample(frac=1).reset_index(drop=True)
		sample_count = len(self.complete_set_df)
		train_split = int(sample_count * 0.80)
		self.train_set_df = self.complete_set_df.head(train_split)
		self.validation_set_df = self.complete_set_df.tail(sample_count - train_split)

	def _normalization(self, im):
		self._subtract_mean(im)

	def _subtract_mean(self, im):
		""" assumes image ordering where channel is after dims """
		im[:,:,:,0] -= 103.939
		im[:,:,:,1] -= 116.779
		im[:,:,:,2] -= 123.68

	def validation_samples_count(self):
		return len(self.validation_set_df.index)

	def training_samples_count(self):
		return len(self.train_set_df.index)

	def trainGen(self, batch_size=8, is_Validation=False):
		if is_Validation:
			train_df = self.validation_set_df
		else:
			train_df = self.train_set_df
		
		#train_df['img'] = train_df['img'].astype(str)

		start = 0
		limit = len(train_df.index)
		
		i = start

		while True:
			if i+1 >= limit:
				i = start
			if i + batch_size > limit:
				end = limit
			else:
				end = i + batch_size

			x_train_from_src = []
			y_train_from_src = []
			for index in range(i, i + batch_size):
				jpg_filename = train_df.loc[index, 'img']
				jpg_img_orig = cv2.imread(self.train_dir + jpg_filename)

				mask_img_orig = cv2.imread(self.train_mask_dir + jpg_filename)

				binary_mask = get_street_mask(mask_img_orig)

				expanded_binary_mask = np.expand_dims(binary_mask, axis=2) # tensorflow expects channels to come after dims

				x_train_from_src.append(jpg_img_orig)
				y_train_from_src.append(expanded_binary_mask)

			x_train_from_src = np.array(x_train_from_src, np.float32)
			y_train_from_src = np.array(y_train_from_src, np.uint8)
			y_train_from_src = y_train_from_src.astype(int)
			#print('x_train_from_src shape', x_train_from_src.shape)
			#print('y_train_from_src shape', y_train_from_src.shape)
			self._normalization(x_train_from_src)
			#x_train_from_src = x_train_from_src.transpose(0,3,1,2) # theano expects channels come before dims
			yield x_train_from_src, y_train_from_src
			i += batch_size
