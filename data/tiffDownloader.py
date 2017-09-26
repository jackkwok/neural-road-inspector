import sys
import urllib
import os.path

def downloadFiles(fileUrlList, overwriteIfExists=False):
	for fileUrl in fileUrlList:
		filename = fileUrl[fileUrl.rfind("/")+1:]
		if overwriteIfExists or not os.path.isfile(filename):
			opener = urllib.URLopener()
			opener.retrieve(fileUrl, filename)
			print('finished downloading file {}:'.format(filename))

def filterListByExtension(fileUrlList, extension):
	result = []
	for fileUrl in fileUrlList:
		#fileExt = fileUrl[fileUrl.rfind(".")+1:]
		if fileUrl.endswith(extension):
			print('adding file {} to queue'.format(fileUrl))
			result.append(fileUrl)
	return result

# execution starts here. command line args processing.
if len(sys.argv) > 1:
	fileWithUrlList = sys.argv[1]
	with open(fileWithUrlList) as f:
		content = f.readlines()
		# remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content] 
		tiffList = filterListByExtension(content, '.tif')
		downloadFiles(tiffList)
else:
	print ('error: required command line argument missing. Syntax: python httpDownloader.py <urls_file>')
	sys.exit(0)
