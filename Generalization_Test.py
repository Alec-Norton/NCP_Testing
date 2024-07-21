import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import glob
import time 
from sklearn.model_selection import train_test_split



import sys

import argparse

def LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.AutoNCP(ncp_size, ncp_output_size, ncp_sparsity_level)
    #Begin constructing layer, starting with input
    
    '''model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape = (None, 8)),
            CfC(wiring),
            tf.keras.layers.Dense(1)
        ]
    )'''
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model

def LTC_FullyConnected(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.FullyConnected(ncp_size, ncp_output_size)
    #Begin constructing layer, starting with input
    
    '''model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape = (None, 8)),
            CfC(wiring),
            tf.keras.layers.Dense(1)
        ]
    )'''
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model



def CNN(input):
    #Algorithm make up is CNN2-a from Trakoolqilaiwan et all.
    #x = tf.keras.layers.Conv2D(32, 3)(input)
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(384, activation = "relu")(x)
    x = tf.keras.layers.Dense(384, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)

#TODO: Load a Time-Series Application

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')



x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])




y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)

x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)

x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)



input = tf.keras.layers.Input(shape = (150, 8))

LTC_NCP_model = LTC_NCP(input, 100, 5, .5)
CNN_model = CNN(input)
LTC_FullyConnected_model = LTC_FullyConnected(input, 100, 5, .2)

base_lr = .02
train_steps = reshape // 64
decay_lr = .66
clipnorm = .9999

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = .33, shuffle = True)


learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )


fc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)
fc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

ncp_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)
ncp_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

cnn_optimizer = tf.keras.optimizers.Adam()
cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

CNN_model.compile(optimizer = cnn_optimizer, loss = cnn_loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())
LTC_NCP_model.compile(optimizer = ncp_optimizer, loss = ncp_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())
LTC_FullyConnected_model.compile(optimizer=fc_optimizer, loss = fc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

LTC_NCP_model.fit(x_train, y_train, 65, 20, 1)

results = LTC_NCP_model.evaluate(x_test, y_test, 64, 1)

print("Accuracy: " + str(results[1]))