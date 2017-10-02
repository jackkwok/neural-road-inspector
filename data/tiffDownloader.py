import sys
import urllib
import os.path

# command line tool to easily download a batch of geotiff files (.tif) from DigitalGlobe Open Data Program: http://digitalglobe.com/opendata/

def downloadFiles(fileUrlList, overwriteIfExists=False):
	for fileUrl in fileUrlList:
		filename = fileUrl[fileUrl.rfind("/")+1:]
		if overwriteIfExists or not os.path.isfile(filename):
			opener = urllib.URLopener()
			opener.retrieve(fileUrl, filename)
			print('downloaded file: {}'.format(filename))

def filterListByExtension(fileUrlList, extension):
	result = []
	for fileUrl in fileUrlList:
		if fileUrl.endswith(extension):
			result.append(fileUrl)
			print('added to queue: {}'.format(fileUrl))
	return result

# execution starts here. command line args processing.
if len(sys.argv) == 2 and sys.argv[1] == '-h':
	print('\n\n   Usage: python httpDownloader.py <urls_file> \n\n   Each URL must be on a new line.')
elif len(sys.argv) > 1:
	fileWithUrlList = sys.argv[1]
	with open(fileWithUrlList) as f:
		content = f.readlines()
		# remove whitespace characters at the end of each line
		content = [x.strip() for x in content] 
		tiffList = filterListByExtension(content, '.tif')
		downloadFiles(tiffList)
else:
	print('error: required command line argument missing. \n\n Syntax: python httpDownloader.py <urls_file> \n\n Each URL must be on a new line.')
	sys.exit(0)
