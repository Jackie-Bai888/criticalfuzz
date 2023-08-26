from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras import backend as K
#from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras.backend import set_session
print(tf.__version__)

def global_average_pooling(x):
    return tf.reduce_mean(x, (1, 2))

def global_average_pooling_shape(input_shape):
    return (input_shape[0], input_shape[3])

def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape

def normal_init(shape, dtype=None):
  initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1)
  # initial = tf.random.truncated_normal(shape, stddev=0.1)
  # return K.variable(initial, dtype=dtype)
  return initializer(shape=shape)

def steering_net():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), kernel_initializer=normal_init, strides=(2,2), name='conv1_1', input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, (5, 5), kernel_initializer=normal_init, strides=(2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, (5, 5), kernel_initializer=normal_init, strides=(2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), kernel_initializer=normal_init, strides=(1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), kernel_initializer=normal_init, strides=(1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, kernel_initializer=normal_init, name="dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(100, kernel_initializer=normal_init,  name="dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(50, kernel_initializer=normal_init, name="dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, kernel_initializer=normal_init, name="dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer=normal_init, name="dense_4"))
    model.add(Lambda(atan_layer, output_shape=atan_layer_shape, name="atan_0"))

    return model

def get_model():
    model = steering_net()
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model

def load_model(path):
    model = steering_net()
    model.load_weights(path)
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model
