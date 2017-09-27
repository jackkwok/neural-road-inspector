from __future__ import division, print_function
import numpy as np

from keras.models import *
from keras.layers import concatenate, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering

class Unet(object):
	def __init__(self, num_channels = 3, img_rows = 256, img_cols = 256):
		"""
		Parameters:
			num_channels: the total number of channels for the data (e.g. for images, it would be 3 for RGB and 4 for RGBA)
			img_rows: number of rows for the image (height)
			img_cols: number of columns for the image (width)

		Limitation:
		For most models:
		row and col dimensions should be multiples of 2.  Otherwise, we will see errors from concatenate layer.
			ValueError: "concat" mode can only concatenate layers with matching output shapes except for the concat axis.
			Layer shapes: [(None, 512, 160, 238), (None, 256, 160, 239)]
		"""
		self.num_channels = num_channels
		self.img_rows = img_rows
		self.img_cols = img_cols

	def downsampling_block(self, input_tensor, filters, padding='valid',
						   batchnorm=False, dropout=0.0):
		_, height, width, _ = K.int_shape(input_tensor)
		assert height % 2 == 0
		assert width % 2 == 0

		x = Conv2D(filters, kernel_size=(3,3), padding=padding,
				   dilation_rate=1)(input_tensor)
		x = BatchNormalization()(x) if batchnorm else x
		x = Activation('relu')(x)
		x = Dropout(dropout)(x) if dropout > 0 else x

		x = Conv2D(filters, kernel_size=(3,3), padding=padding, dilation_rate=2)(x)
		x = BatchNormalization()(x) if batchnorm else x
		x = Activation('relu')(x)
		x = Dropout(dropout)(x) if dropout > 0 else x

		return MaxPooling2D(pool_size=(2,2))(x), x

	def upsampling_block(self, input_tensor, skip_tensor, filters, padding='valid',
						 batchnorm=False, dropout=0.0):
		x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

		# compute amount of cropping needed for skip_tensor
		_, x_height, x_width, _ = K.int_shape(x)
		_, s_height, s_width, _ = K.int_shape(skip_tensor)
		h_crop = s_height - x_height
		w_crop = s_width - x_width
		assert h_crop >= 0
		assert w_crop >= 0
		if h_crop == 0 and w_crop == 0:
			y = skip_tensor
		else:
			cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
			y = Cropping2D(cropping=cropping)(skip_tensor)

		x = Concatenate()([x, y])

		# no dilation in upsampling convolutions
		x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
		x = BatchNormalization()(x) if batchnorm else x
		x = Activation('relu')(x)
		x = Dropout(dropout)(x) if dropout > 0 else x

		x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
		x = BatchNormalization()(x) if batchnorm else x
		x = Activation('relu')(x)
		x = Dropout(dropout)(x) if dropout > 0 else x

		return x

	def dilated_unet(self, classes=1, features=32, depth=4,
					 temperature=1.0, padding='same', batchnorm=False,
					 dropout=0.0, dilation_layers=5):
		"""
		Generate `dilated U-Net' model where the convolutions in the encoding and
		bottleneck are replaced by dilated convolutions. The second convolution in
		pair at a given scale in the encoder is dilated by 2. The number of
		dilation layers in the innermost bottleneck is controlled by the
		`dilation_layers' parameter -- this is the `context module' proposed by Yu,
		Koltun 2016 in "Multi-scale Context Aggregation by Dilated Convolutions"

		Arbitrary number of input channels and output classes are supported.

		Arguments:
		  classes - number of output classes (2 in paper)
		  features - number of output features for first convolution (64 in paper)
			  Number of features double after each down sampling block
		  depth  - number of downsampling operations (4 in paper)
		  padding - 'valid' (used in paper) or 'same'
		  batchnorm - include batch normalization layers before activations
		  dropout - fraction of units to dropout, 0 to keep all units
		  dilation_layers - number of dilated convolutions in innermost bottleneck

		Output:
		  Dilated U-Net model expecting input shape (height, width, maps) and
		  generates output with shape (output_height, output_width, classes).
		  If padding is 'same', then output_height = height and
		  output_width = width.

		"""
		x = Input(shape=(self.img_rows, self.img_cols, self.num_channels))
		inputs = x

		skips = []
		for i in range(depth):
			x, x0 = self.downsampling_block(x, features, padding,
									   batchnorm, dropout)
			skips.append(x0)
			features *= 2

		dilation_rate = 1
		for n in range(dilation_layers):
			x = Conv2D(filters=features, kernel_size=(3,3), padding=padding,
					   dilation_rate=dilation_rate)(x)
			x = BatchNormalization()(x) if batchnorm else x
			x = Activation('relu')(x)
			x = Dropout(dropout)(x) if dropout > 0 else x
			dilation_rate *= 2

		for i in reversed(range(depth)):
			features //= 2
			x = self.upsampling_block(x, skips[i], features, padding,
								 batchnorm, dropout)

		x = Conv2D(filters=classes, kernel_size=(1,1))(x)

		logits = Lambda(lambda z: z/temperature)(x)
		probabilities = Activation('sigmoid')(logits)
		return Model(inputs=inputs, outputs=probabilities)

	def get_crop_shape(self, target, refer):
		"""
		get_crop_shape allows model input dimension to be any arbitrary integers.
			
			Theano ordering where height is at index 2 and width is at index 3.
			Tensorflow sould have height at index 1 and width at index 2
		"""
		width_index = 2 # TF ordering
		height_index = 1 # TF ordering

		# width
		cw = (keras.int_shape(target)[width_index] - keras.int_shape(refer)[width_index])
		assert (cw >= 0)
		if cw % 2 != 0:
			cw1, cw2 = int(cw/2), int(cw/2) + 1
		else:
			cw1, cw2 = int(cw/2), int(cw/2)
		# height
		ch = (keras.int_shape(target)[height_index] - keras.int_shape(refer)[height_index])
		assert (ch >= 0)
		if ch % 2 != 0:
			ch1, ch2 = int(ch/2), int(ch/2) + 1
		else:
			ch1, ch2 = int(ch/2), int(ch/2)

		return (ch1, ch2), (cw1, cw2)

	# original work on u-net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
	# note: original u-net use padding='valid' instead of 'same'
	# code borrowed from: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/51dc5a1b2b77ac5b75dda77f3577c7c6bcf2b2a9/train.py
	# they use a differnent lose function
	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool4)
		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv5)

		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3) # concat_axis=3 for Tensorflow vs 1 for theano
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(up6)
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(up7)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(up8)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(up9)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv9)

		conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

		model = Model(inputs=[inputs], outputs=[conv10])
		return model

	def get_unet_dilated(self):
		"""
			Generate `dilated U-Net' model where the convolutions in the encoding and
			bottleneck are replaced by dilated convolutions. The second convolution in
			pair at a given scale in the encoder is dilated by 2.
		"""
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(inputs)
		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool1)
		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool2)
		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool3)
		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool4)
		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv5)

		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3) # concat_axis=3 for Tensorflow vs 1 for theano
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(up6)
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(up7)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(up8)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(up9)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv9)

		conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

		model = Model(inputs=[inputs], outputs=[conv10])
		return model

	def get_unet_dilated_d6(self):
		"""
			Generate `dilated U-Net' model where the convolutions in the encoding and
			bottleneck are replaced by dilated convolutions. The second convolution in
			pair at a given scale in the encoder is dilated by 2.
		"""
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv0 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(inputs)
		conv0 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv0)
		pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool0)
		conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool1)
		conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool2)
		conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool3)
		conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", dilation_rate=(1, 1))(pool4)
		conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", dilation_rate=(2, 2))(conv5)

		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3) # concat_axis=3 for Tensorflow vs 1 for theano
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(up6)
		conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(up7)
		conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(up8)
		conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(up9)
		conv9 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv9)

		up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv0], axis=3)
		conv10 = Conv2D(32, (3, 3), padding="same", activation="relu")(up10)
		conv10 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv10)

		conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

		model = Model(inputs=[inputs], outputs=[conv11])
		return model

	def get_unet_level_7(self):
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv0 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
		conv0 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv0)
		pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

		conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool0)
		conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
		conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
		conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
		conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
		conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
		pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)
		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
		conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
		conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
		conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
		conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
		conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
		conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

		up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
		conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(up10)
		conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)

		up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
		conv11 = Conv2D(16, (3, 3), activation='relu', padding='same')(up11)
		conv11 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv11)

		up12 = concatenate([UpSampling2D(size=(2, 2))(conv11), conv0], axis=3)
		conv12 = Conv2D(8, (3, 3), activation='relu', padding='same')(up12)
		conv12 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv12)

		conv13 = Conv2D(1, (1, 1), activation='sigmoid')(conv12)

		model = Model(input=inputs, output=conv13)

		return model

	def get_unet_level_8(self):
		"""
			Apply Cropping2D similar to : https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py 
			Cropping before concatenate to allow arbitrary image dimensions 
		"""
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv0 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
		conv0 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv0)
		pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

		conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool0)
		conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
		conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
		conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
		conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
		conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
		pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool5)
		conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
		pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

		# bottom of the UNet
		conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool6)
		conv7 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv7)

		up_conv7 = UpSampling2D(size=(2, 2))(conv7)
		ch, cw = self.get_crop_shape(conv6, up_conv7)
		crop_conv6 = Cropping2D(cropping=(ch,cw))(conv6)
		up8 = concatenate([up_conv7, crop_conv6], axis=3)
		conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(up8)
		conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)

		up_conv8 = UpSampling2D(size=(2, 2))(conv8)
		ch, cw = self.get_crop_shape(conv5, up_conv8)
		crop_conv5 = Cropping2D(cropping=(ch,cw))(conv5)
		up9 = concatenate([up_conv8, crop_conv5], axis=3)
		conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(up9)
		conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv9)

		up_conv9 = UpSampling2D(size=(2, 2))(conv9)
		ch, cw = self.get_crop_shape(conv4, up_conv9)
		crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)		
		up10 = concatenate([up_conv9, crop_conv4], axis=3)
		conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(up10)
		conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv10)

		up_conv10 = UpSampling2D(size=(2, 2))(conv10)
		ch, cw = self.get_crop_shape(conv3, up_conv10)
		crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
		up11 = concatenate([up_conv10, crop_conv3], axis=3)
		conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(up11)
		conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)

		up_conv11 = UpSampling2D(size=(2, 2))(conv11)
		ch, cw = self.get_crop_shape(conv2, up_conv11)
		crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
		up12 = concatenate([up_conv11, crop_conv2], axis=3)
		conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(up12)
		conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv12)

		up_conv12 = UpSampling2D(size=(2, 2))(conv12)
		ch, cw = self.get_crop_shape(conv1, up_conv12)
		crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
		up13 = concatenate([up_conv12, crop_conv1], axis=3)
		conv13 = Conv2D(16, (3, 3), activation='relu', padding='same')(up13)
		conv13 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv13)

		up_conv13 = UpSampling2D(size=(2, 2))(conv13)
		ch, cw = self.get_crop_shape(conv0, up_conv13)
		crop_conv0 = Cropping2D(cropping=(ch,cw))(conv0)
		up14 = concatenate([up_conv13, crop_conv0], axis=3)
		conv14 = Conv2D(8, (3, 3), activation='relu', padding='same')(up14)
		conv14 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv14)

		# Must add padding to match Input dimenions. Otherwise:
		# ValueError: GpuElemwise. Input dimension mis-match. Input 1 (indices start at 0) has shape[0] == 2293760, but the output's size on that axis is 2455040.
		ch, cw = self.get_crop_shape(inputs, conv14)
		padding14 = ZeroPadding2D(padding=(ch[0], ch[1], cw[0], cw[1]))(conv14)  # (top_pad, bottom_pad, left_pad, right_pad)
		conv15 = Conv2D(1, (1, 1), activation='sigmoid')(padding14)

		model = Model(input=inputs, output=conv15)

		return model

	def get_unet_mini(self):
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
		conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool1)
		conv2 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool2)
		conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool3)
		conv4 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool4)
		conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv5)

		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3) # concat_axis=3 for Tensorflow vs 1 for theano
		conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(up6)
		conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
		conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(up7)
		conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
		conv8 = Conv2D(32, (3, 3), padding="same", activation="relu")(up8)
		conv8 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
		conv9 = Conv2D(16, (3, 3), padding="same", activation="relu")(up9)
		conv9 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv9)

		conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

		model = Model(inputs=[inputs], outputs=[conv10])
		return model

	def get_unet_mini_bn(self):
		inputs = Input((self.img_rows, self.img_cols, self.num_channels))

		conv1 = Conv2D(16, 3, 3, padding='same')(inputs)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		conv1 = Conv2D(16, 3, 3, padding='same')(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(32, 3, 3, padding='same')(pool1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		conv2 = Conv2D(32, 3, 3, padding='same')(conv2)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(64, 3, 3, padding='same')(pool2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		conv3 = Conv2D(64, 3, 3, padding='same')(conv3)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(128, 3, 3, padding='same')(pool3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		conv4 = Conv2D(128, 3, 3, padding='same')(conv4)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(256, 3, 3, padding='same')(pool4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		conv5 = Conv2D(256, 3, 3, padding='same')(conv5)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)

		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
		conv6 = Conv2D(128, 3, 3, padding='same')(up6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		conv6 = Conv2D(128, 3, 3, padding='same')(conv6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)

		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
		conv7 = Conv2D(64, 3, 3, padding='same')(up7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		conv7 = Conv2D(64, 3, 3, padding='same')(conv7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)

		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
		conv8 = Conv2D(32, 3, 3, padding='same')(up8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		conv8 = Conv2D(32, 3, 3, padding='same')(conv8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)

		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
		conv9 = Conv2D(16, 3, 3, padding='same')(up9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)
		conv9 = Conv2D(16, 3, 3, padding='same')(conv9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Activation('relu')(conv9)

		conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

		model = Model(input=inputs, output=conv10)

		return model
