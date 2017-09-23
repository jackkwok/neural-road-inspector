import numpy as np
import pandas as pd
import cv2
import os
from maskprocessor import *

from augmentation import *

class CustomImgGenerator(object):
	""" 
		A Custom Image Generator that generate
	    training set and validation set with a 8:2 split. 
	"""
	def __init__(self, training_x_dir, training_y_dir, csv_path):
		self.train_dir = training_x_dir
		self.train_mask_dir = training_y_dir
		self.complete_set_df = pd.read_csv(csv_path)
		# shuffle everything in the training set. must reset the dataframe index.
		self.complete_set_df = self.complete_set_df.sample(frac=1).reset_index(drop=True)
		sample_count = len(self.complete_set_df)
		train_split = int(sample_count * 0.80)
		self.train_set_df = self.complete_set_df.head(train_split).reset_index(drop=True)
		self.validation_set_df = self.complete_set_df.tail(sample_count - train_split).reset_index(drop=True)

	def _normalization(self, im):
		self._subtract_mean(im)

	def _subtract_mean(self, im):
		""" assumes image ordering where channel is after dims """
		im = im.astype(float)
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

		limit = len(train_df.index)
		#print('size of data set', len(train_df.index))
		
		i = 0

		while True:
			if i >= limit:
				i = 0
			if i + batch_size > limit:
				end = limit
			else:
				end = i + batch_size

			x_train_from_src = []
			y_train_from_src = []
			for index in range(i, end):
				jpg_filename = train_df.loc[index, 'img']
				jpg_img_orig = cv2.imread(self.train_dir + jpg_filename)

				png_filename = os.path.splitext(jpg_filename)[0] + '.png'
				mask_img_orig = cv2.imread(self.train_mask_dir + png_filename)

				binary_mask = get_street_mask(mask_img_orig)

				expanded_binary_mask = np.expand_dims(binary_mask, axis=2) # tensorflow expects channels to come after dims

				if not is_Validation:
					jpg_img_orig = random_gaussian_blur(jpg_img_orig)

				x_train_from_src.append(jpg_img_orig)
				y_train_from_src.append(expanded_binary_mask)

			x_train_from_src = np.array(x_train_from_src, np.float32)
			y_train_from_src = np.array(y_train_from_src, np.uint8)
			y_train_from_src = y_train_from_src.astype(int)
			#print('x_train_from_src shape', x_train_from_src.shape)
			#print('y_train_from_src shape', y_train_from_src.shape)
			if not is_Validation:
			 	x_train_from_src, y_train_from_src = apply_augment_sequence(x_train_from_src, y_train_from_src)
			self._normalization(x_train_from_src)
			yield x_train_from_src, y_train_from_src
			i += batch_size
