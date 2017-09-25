
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from datetime import datetime
import time
import sys
import configparser
import json
import pickle

import matplotlib.pyplot as plt
#%matplotlib inline

from unet.generator import *
from unet.loss import *
from unet.maskprocessor import *
from unet.visualization import *
from unet.modelfactory import *


# This notebook trains the Road Segmentation model.  The exported .py script takes in the config filename via a command line parameter.  To run this notebook directly in jupyter notebook, please manually set config_file to point to a configuration file (e.g. cfg/default.cfg).

# In[2]:

# command line args processing "python RoadSegmentor.py cfg/your_config.cfg"
if len(sys.argv) > 1 and '.cfg' in sys.argv[1]:
    config_file = sys.argv[1]
else:
    config_file = 'cfg/default.cfg'
    #print('missing argument. please provide config file as argument. syntax: python RoadSegmentor.py <config_file>')
    #exit(0)

print('reading configurations from config file: {}'.format(config_file))

settings = configparser.ConfigParser()
settings.read(config_file)

x_data_dir = settings.get('data', 'train_x_dir')
y_data_dir = settings.get('data', 'train_y_dir')
print('x_data_dir: {}'.format(x_data_dir))
print('y_data_dir: {}'.format(y_data_dir))

data_csv_path = settings.get('data', 'train_list_csv')

print('model configuration options:', settings.options("model"))

model_dir = settings.get('model', 'model_dir')
print('model_dir: {}'.format(model_dir))

timestr = time.strftime("%Y%m%d-%H%M%S")

model_id = settings.get('model', 'id')
print('model: {}'.format(model_id))

optimizer_label = 'Adam' # default

if settings.has_option('model', 'optimizer'):
    optimizer_label = settings.get('model', 'optimizer')
    
if settings.has_option('model', 'source'):
    model_file = settings.get('model', 'source')
    print('model_file: {}'.format(model_file))
else:
    model_file = None

learning_rate = settings.getfloat('model', 'learning_rate')
max_number_epoch = settings.getint('model', 'max_epoch')
print('learning rate: {}'.format(learning_rate))
print('max epoch: {}'.format(max_number_epoch))

min_learning_rate = 0.000001
if settings.has_option('model', 'min_learning_rate'):
    min_learning_rate = settings.getfloat('model', 'min_learning_rate')
print('minimum learning rate: {}'.format(min_learning_rate))

lr_reduction_factor = 0.1
if settings.has_option('model', 'lr_reduction_factor'):
    lr_reduction_factor = settings.getfloat('model', 'lr_reduction_factor')
print('lr_reduction_factor: {}'.format(lr_reduction_factor))

batch_size = settings.getint('model', 'batch_size') 
print('batch size: {}'.format(batch_size))


# In[3]:

img_gen = CustomImgGenerator(x_data_dir, y_data_dir, data_csv_path)

train_gen = img_gen.trainGen(batch_size=batch_size, is_Validation=False)

validation_gen = img_gen.trainGen(batch_size=batch_size, is_Validation=True)


# In[4]:

timestr = time.strftime("%Y%m%d-%H%M%S")
model_filename = model_dir + '{}-{}.hdf5'.format(model_id, timestr)
print('model checkpoint file path: {}'.format(model_filename))

# Early stopping prevents overfitting on training data
# Make sure the patience value for EarlyStopping > patience value for ReduceLROnPlateau. 
# Otherwise ReduceLROnPlateau will never be called.
early_stop = EarlyStopping(monitor='val_loss',
                           patience=3,
                           min_delta=0, 
                           verbose=1,
                           mode='auto')

model_checkpoint = ModelCheckpoint(model_filename,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)

reduceLR = ReduceLROnPlateau(monitor='val_loss',
                             factor=lr_reduction_factor,
                             patience=2,
                             verbose=1,
                             min_lr=min_learning_rate,
                             epsilon=1e-4)


# In[10]:

training_start_time = datetime.now()

number_validations = img_gen.validation_samples_count()

samples_per_epoch = img_gen.training_samples_count()

modelFactory = ModelFactory(num_channels = 3, 
                            img_rows = 512,
                            img_cols = 512)

if model_file is not None:
    model = load_model(model_dir + model_file,
                  custom_objects={'dice_coef_loss': dice_coef_loss, 
                                  'dice_coef': dice_coef, 
                                  'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss})
else:
    model = modelFactory.get_model(model_id)

print(model.summary())

if optimizer_label == 'Adam':
    optimizer = Adam(lr = learning_rate)
elif optimizer_label == 'RMSprop':
    optimizer = RMSprop(lr = learning_rate)
else:
    raise ValueError('unsupported optimizer: {}'.format(optimizer_label))

model.compile(optimizer = optimizer,
              loss = dice_coef_loss,
              metrics = ['accuracy', dice_coef])


# In[ ]:

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=np.ceil(float(samples_per_epoch) / float(batch_size)),
                              validation_data=validation_gen,
                              validation_steps=np.ceil(float(number_validations) / float(batch_size)),
                              epochs=max_number_epoch,
                              verbose=1,
                              callbacks=[model_checkpoint, early_stop, reduceLR])

time_spent_trianing = datetime.now() - training_start_time
print('model training complete. time spent: {}'.format(time_spent_trianing))


# In[ ]:

print(history.history)

historyFilePath = model_dir + '{}-{}-train-history.png'.format(model_id, timestr)
trainingHistoryPlot(model_id + timestr, historyFilePath, history.history)

pickleFilePath = model_dir + '{}-{}-history-dict.pickle'.format(model_id, timestr)
with open(pickleFilePath, 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

