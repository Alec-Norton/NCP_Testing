import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
import csv
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import glob
import time 
from sklearn.model_selection import train_test_split
import keras_tuner as kt


import sys

import argparse

keras = tf.keras
print("\n")
print("Loading Models: ")
CNN_model = keras.models.load_model('CNN_Model.keras')
LTC_NCP_model = keras.models.load_model('LTC_NCP_Model.keras')

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts\*.csv')


x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])




#csv_file = pd.read_csv('size_30sec_150ts_stride_03ts\sub_1.csv')
#x_train = csv_file.copy()

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



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = .33, shuffle = True)
noise_x = []
CNN_accuracy = []
LTC_NCP_accuracy = []

print("Noise_Testing: ")

noise_x.append(0)
CNN_results = CNN_model.evaluate(x_valid, y_valid, verbose = 1)
LTC_NCP_results = LTC_NCP_model.evaluate(x_valid, y_valid, verbose = 1)
CNN_accuracy.append(CNN_results[1])
LTC_NCP_accuracy.append(LTC_NCP_results[1])

for i in range(0, .5, .01):
    noise_copy = x_valid + np.random.normal(0, i, x_valid.shape)
    CNN_results = CNN_model.evaluate(x_valid, y_valid, verbose = 0)
    LTC_NCP_results = LTC_NCP_model.evaluate(x_valid, y_valid, verbose = 0)
    noise_x.append(i)
    CNN_accuracy.append(CNN_results[1])
    LTC_NCP_accuracy.append(LTC_NCP_results[1])



plt.plot(noise_x, CNN_accuracy, label = "CNN", linestyle = ":")
plt.plot(noise_x, LTC_NCP_accuracy, label = "LTC_NCP", linestyle = ":")






