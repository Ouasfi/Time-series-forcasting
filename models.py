import numpy as np
import pandas as pd 
import tensorflow as tf
from keras.layers import *
from keras import Sequential
import keras

from tensorflow.keras import layers


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def train_test_split(series):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(241).prefetch(1)


def conv1d_bolck( filters= [3], kernels = [4], stride = 1,  activation = 'relu', 
                  dropout_rate = 0.3):
    layers = []
    for (f, k) in zip(filters, kernels):
        
        layers.append(tf.keras.layers.BatchNormalization( momentum=0.1, epsilon=1e-05))
        layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.Conv1D(filters=f, kernel_size= k, strides =stride, padding = "causal", activation = activation))
       # layers.append(tf.keras.layers.MaxPooling1D())
    
    return layers



def lstm_block( activation = 'relu', units = [100,100, 1], momentum=0.1, epsilon=1e-05,dropout_rate = 0.3 ):
    layers = []
    for n_units in units:
        
        layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.LSTM(units=n_units,  activation=activation, return_sequences=True ))
    
    return layers

def dense_block(activation = 'relu', units = [100,100, 1], momentum=0.1, epsilon=1e-05, dropout_rate = 0.3):
    
    
    layers = [tf.keras.layers.BatchNormalization( momentum=momentum, epsilon=epsilon)]
    
    for n_units in units[:-1]:
        
        layers.append(tf.keras.layers.Dropout(dropout_rate))
        layers.append(tf.keras.layers.Dense(units=n_units,  activation=activation))
        layers.append(tf.keras.layers.BatchNormalization (momentum=momentum, epsilon=epsilon))
    layers.append(tf.keras.layers.Dense(units=units[-1]))

    return layers



class ScaleLayer(layers.Layer):
    def __init__(self):
        super(ScaleLayer, self).__init__()

        w_init = tf.random_normal_initializer()

        self.scale = tf.Variable(initial_value=w_init(shape=(1,),
                                              dtype='float32'),
                         trainable=True)


    def call(self, inputs):
        return inputs * self.scale
    
    def get_config(self):
        return {'scale': self.scale}


