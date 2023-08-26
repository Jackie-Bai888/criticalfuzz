from __future__ import absolute_import
from __future__ import print_function
import os


os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'


import keras
import keras.models as models
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape

from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
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

def steering_net():
    # model start here
    model = models.Sequential()
    model.add(Convolution2D(16, (3, 3), input_shape=(32, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    return model



def get_model():
    model = steering_net()
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error', metrics=['mse', 'accuracy'])
    return model









