from __future__ import absolute_import
from __future__ import print_function
import os


os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'


import keras
import keras.models as models

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
import sklearn.metrics as metrics

from keras.callbacks import ModelCheckpoint

import cv2
import numpy as np
import dav1_model
import json
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()


ndata = 0
imgsize = 128
# frame size
nrows = 128
ncols = 128


data_path = '/common/hahabai/data/autonomous_driving_dataset/dav2_data/'
label_dir = {}
print('---start get label---')
data_df = pd.read_csv(os.path.join(data_path, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering',
                                 'throttle', 'reverse', 'speed'])

print('---finish get label---')
img_path = data_path+'IMG'
imgs = []
targets = []
print('---start get image---')
for index, row_name in enumerate(data_df['center']):
    img_name = row_name.split('\\')[-1]
    img = cv2.imread(os.path.join(img_path, img_name))
    imgs.append(cv2.resize(img, (128, 128)))
    targets.append(data_df['steering'])
'''
for img_name in os.listdir(img_path):
    if 'center' in img_name:
        print('img_name', img_name)
        img = cv2.imread(os.path.join(img_path,img_name))
        imgs.append(cv2.resize(img, (128, 128)))
        targets.append(label_dir[img_name])
'''
print('---finish get image---')
imgs = np.array(imgs)
targets = np.array(targets)
np.save("uda_imgs.npy",imgs)
np.save("uda_labels.npy",targets)
# imgs = np.load('imgs.npy')
# targets = np.load("labels.npy")


# imgs = np.load('data/imgs_128.npz')['arr_0']
# targets = np.load('data/targets_128.npz')['arr_0']

idx = np.arange(0,imgs.shape[0])
idx = np.random.permutation(idx)
imgs = imgs[idx,:,:,:]
targets = targets[idx]#* scipy.pi / 180
print(np.max(targets), np.min(targets))



# load the model:
model = dav1_model.get_model()


# checkpoint
filepath="weights/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 25
batch_size = 64

model.fit(imgs, targets, callbacks=callbacks_list,
	batch_size =batch_size, epochs=nb_epoch, verbose=1,
	validation_split=0.1,shuffle=True)


model.save_weights('weights/model_basic_weight.hdf5')
