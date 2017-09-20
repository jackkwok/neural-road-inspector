import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

def trainingHistoryPlot(file_path, history):
	""" Plot training stats over time """
	fig = plt.figure(figsize=(20, 10))

	# history for accuracy
	subplot1 = fig.add_subplot(121)
	subplot1.plot(history['acc'])
	subplot1.plot(history['val_acc'])
	subplot1.set_title('accuracy')
	subplot1.set_ylabel('accuracy')
	subplot1.set_xlabel('epoch')
	subplot1.legend(['train', 'val'], loc='upper left')
	subplot1.grid()

	# history for loss
	subplot2 = fig.add_subplot(122)
	subplot2.plot(history['loss'])
	subplot2.plot(history['val_loss'])
	subplot2.set_title('model loss')
	subplot2.set_ylabel('loss')
	subplot2.set_xlabel('epoch')
	subplot2.legend(['train', 'val'], loc='upper left')
	subplot2.grid()

	fig.savefig(file_path)
