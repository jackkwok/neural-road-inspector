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

def _image_file_list(dir_path):
	""" limitation: the images files must have an image extension: webp, jpg, png, or jpeg """
	result = []
	for root, dirs, files in os.walk(dir_path):
		for file in files:
			if file.endswith('.webp') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
				result.append(os.path.join(root, file))
	return result

def _out_file_list(input_file_list, in_dir, out_dir):
	out_dir = os.path.join(out_dir, '') # add OS-indepedent slash
	in_dir = os.path.join(in_dir, '') # add OS-indepedent slash

	result = []
	for file in input_file_list:
		result.append(file.replace(in_dir, out_dir, 1))
	return result

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def genRoadMask(img_path, out_dir, model_path, is_directory = False):
	"""Given an input image and model, generate and save the Road Mask image to the out_dir"""
	model = _get_model(model_path)

	if is_directory:
		filelist = _image_file_list(img_path)
		output_filelist = _out_file_list(filelist, img_path, out_dir)
		x = []
		for file in filelist:
			print(file)
			img = normalize_img(file, resize=True)
			print(img.shape)
			x.append(img)
		x = np.array(x, np.float32)
		batch_size = 4
	else:
		img = normalize_img(img_path, resize=True)
		x = np.expand_dims(img, axis=0)
		batch_size = 1
	
	y = model.predict(x, batch_size=batch_size, verbose=1)
	#print('y shape', y.shape) # ('y shape', (1, 512, 512, 1))
	
	if is_directory:
		mask = (y > 0.5) # model output are floats and need to be converted to boolean
		mask.dtype = 'uint8'
		mask[mask == 1] = 255

		for index, out_file in enumerate(output_filelist):
			makedirs(os.path.dirname(out_file))
			cv2.imwrite(out_file, mask[index])
	else:
		mask = (y[0] > 0.5) # model output are floats and need to be converted to boolean
		mask.dtype='uint8'
		mask[mask == 1] = 255
		#print('mask shape', mask.shape)
		img_filename = ntpath.basename(img_path)

		file_no_ext, file_ext = os.path.splitext(img_path)
		if file_ext == '':
			img_filename = img_filename + '.jpg'

		cv2.imwrite(out_dir + img_filename, mask)

# execution starts here. command line args processing.
if len(sys.argv) > 3:
	input_file_path = sys.argv[1]
	# input_file_path is allowed to be a single file or a directory
	if os.path.isdir(input_file_path):
		is_directory = True;
	elif os.path.isfile(input_file_path):
		is_directory = False;
	else:
		print('error: input image path {} does not exist', input_file_path)
		sys.exit(0)
	output_dir = sys.argv[2]
	output_dir = os.path.join(output_dir, '') # add OS-indepedent slash
	makedirs(output_dir)
	model_path = sys.argv[3]
	if not os.path.isfile(model_path):
		print('error: model file {} does not exist', model_path)
		sys.exit(0)
	genRoadMask(input_file_path, output_dir, model_path, is_directory = is_directory)
elif len(sys.argv) == 2 and sys.argv[1] == '-h':
	print('******************** \n\n Usage: \n\n python roadSegmentationMaskGen.py <satellite_images_dir> <output_dir> <keras_model_filepath>\n\n   Limitions: input images must have 3 channels only (images with alpha channel not supported) \n\n********************')
else:
	print ('error: required command line argument missing. Syntax: python roadSegmentationMaskGen.py <satellite_images_dir> <output_dir> <keras_model_filepath>')
	sys.exit(0)


