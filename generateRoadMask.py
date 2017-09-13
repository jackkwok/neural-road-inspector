import sys
import os
import ntpath
import cv2

from keras.models import *

from unet.unet import *
from unet.generator import *
from unet.loss import *
from unet.maskprocessor import *
from unet.normalization import *

def _get_model(model_path):
	model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss,
												   'dice_coef': dice_coef,
												   'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss})
	return model

def genRoadMask(img_path, out_dir, model_path):
	"""Given an input image and model, generate and save the Road Mask image to the out_dir"""
	model = _get_model(model_path)
	img = normalize_img(img_path, resize=True)
	x = np.expand_dims(img, axis=0)
	y = model.predict(x, batch_size=1, verbose=1)
	#print('y shape', y.shape) # ('y shape', (1, 512, 512, 1))
	mask = (y[0] > 0.5) # model output are floats and need to be converted to boolean
	mask.dtype='uint8'
	mask[mask==1] = 255
	#print('mask shape', mask.shape)
	img_filename = ntpath.basename(img_path)

	file_no_ext, file_ext = os.path.splitext(img_path)
	if file_ext == '':
		img_filename = img_filename + '.jpg'

	cv2.imwrite(out_dir + img_filename, mask)

# execution starts here. command line args processing.
if len(sys.argv) > 3:
	input_file_path = sys.argv[1]
	if not os.path.isfile(input_file_path):
		print('error: input image file {} does not exist', input_file_path)
		sys.exit(0)
	output_dir = sys.argv[2]
	output_dir = os.path.join(output_dir, '') # add OS-indepedent slash
	model_path = sys.argv[3]
	if not os.path.isfile(model_path):
		print('error: model file {} does not exist', model_path)
		sys.exit(0)
	genRoadMask(input_file_path, output_dir, model_path)
else:
	print ('error: required command line argument missing. e.g. python generateRoadMask.py input.png /home/output/ /home/models/model.hdf5')
	sys.exit(0)


