from __future__ import print_function
from mapbox import Uploader
from time import sleep
from random import randint
import os
from tqdm import tqdm

#filepath = '/Users/jkwok/Documents/Insight/digital_globe/3021113.tif'
#filepath = '/Users/jkwok/Documents/Insight/jupyter/data/3020110.tif'
filepath = '/Users/jkwok/Documents/Insight/jupyter/data/3020121.tif'

filename = filepath.split('/')[-1]

mapid = 'digitalglobe_harvey_{}'.format(filename.replace('.', '_'))

totalBytes = os.path.getsize(filepath)

class TqdmCustom(tqdm):
	def progressCallback(numberOfBytes, wtf):
		self.total = totalBytes
		self.update(numberOfBytes)

progressBar = TqdmCustom(unit='B', unit_scale=True, miniters=1, desc=filename)

service = Uploader()

service.session.params['access_token'] = os.environ['PRIVATE_MAPBOX_ACCESS_TOKEN']

with open(filepath, 'rb') as src:
	upload_resp = service.upload(src, mapid, callback=None) #progressBar.progressCallback)

if upload_resp.status_code == 422:
	for i in range(5):
		sleep(5)
		with open(filepath, 'rb') as src:
			upload_resp = service.upload(src, mapid)
		if upload_resp.status_code != 422:
			break

print(upload_resp.status_code)

upload_id = upload_resp.json()['id']
for i in range(5):
	status_resp = service.status(upload_id).json()
	if status_resp['complete']:
		break
	sleep(5)

print(mapid in status_resp['tileset'])




