from __future__ import absolute_import
from __future__ import print_function
import os


os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'


import keras
import keras.models as models
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization,Input
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D
from tensorflow.keras.optimizers import Adam
import sklearn.metrics as metrics

import cv2
import numpy as np
import json

# from Dan Does Data VLOG
import math
import h5py
import glob
import scipy
from scipy import misc

import matplotlib.pyplot as plt
plt.ion()


ndata = 0
imgsize = 128
# frame size
nrows = 128
ncols = 128

def steering_net():
    # model start here
    model = Sequential()

    model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(3, nrows,ncols)))

    model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))



def get_model():
    model = steering_net()
    adam = Adam(lr=0.0001)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse', 'accuracy'])
    return model









