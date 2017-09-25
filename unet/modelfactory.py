from unet import *
from tiramisu import *

class ModelFactory(object):
	def __init__(self, num_channels = 3, img_rows = 256, img_cols = 256):
		"""
		Parameters:
			num_channels: the total number of channels for the data (e.g. for images, it would be 3 for RGB and 4 for RGBA)
			img_rows: number of rows for the image (height)
			img_cols: number of columns for the image (width)

		Limitation:
			See tiramisu.py and unet.py
		"""
		self.num_channels = num_channels
		self.img_rows = img_rows
		self.img_cols = img_cols

	def get_model(self, model_id):
		if (model_id == 'Tiramisu') :
			tiramisu = Tiramisu(self.num_channels, self.img_rows, self.img_cols)
			return tiramisu.get_tiramisu()
		else:
			unet = Unet(self.num_channels, self.img_rows, self.img_cols)
			model_dict = {
				'Unet': unet.get_unet,
				'Unet_Mini': unet.get_unet_mini,
				'Unet_Level7': unet.get_unet_level_7,
				'Unet_Level8': unet.get_unet_level_8,
				'Unet_Dilated': unet.get_unet_dilated,
				'Unet_Dilated_D6': unet.get_unet_dilated_d6,
				'Dilated_Unet': unet.dilated_unet,
			}
			return model_dict[model_id]()
