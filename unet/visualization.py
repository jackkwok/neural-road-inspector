import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

def trainingHistoryPlot(title, file_path, history):
	"""
		Save a plot of training stats (training/validation accuricy and loss) over time 

		Parameters:
			title: title shown on the graph
			file_path: file path to save the generated graph
			history: dictionary from the Keras History object
	"""
	plt.rcParams.update({'font.size': 26})
	fig = plt.figure(figsize=(20, 10))

	plt.title(title)

	# history for accuracy
	subplot1 = fig.add_subplot(121)
	subplot1.plot(history['acc'])
	subplot1.plot(history['val_acc'])
	subplot1.set_ylabel('accuracy')
	subplot1.set_xlabel('epoch')
	subplot1.legend(['train', 'validation'], loc='lower right')
	subplot1.grid()

	# history for loss
	subplot2 = fig.add_subplot(122)
	subplot2.plot(history['loss'])
	subplot2.plot(history['val_loss'])
	subplot2.set_ylabel('loss')
	subplot2.set_xlabel('epoch')
	subplot2.legend(['train', 'validation'], loc='upper right')
	subplot2.grid()

	fig.savefig(file_path)

def plotValLoss(title, file_path, history):
	"""
		Save a plot of training stats (training/validation loss) over time 

		Parameters:
			model_id: used for showing as title of the graph
			file_path: file path to save the generated graph
			history: dictionary from the Keras History object
	"""
	plt.rcParams.update({'font.size': 34})
	fig = plt.figure(figsize=(20, 20))
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title(title)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.grid()
	fig.savefig(file_path)


