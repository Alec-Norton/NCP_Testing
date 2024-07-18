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
    x = keras.layers.Flatten()(x)
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
    x = keras.layers.Flatten()(x)
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

LTC_NCP_accuracy = []
CNN_accuracy = []
LTC_FC_accuracy = []
split_x = []

#'''

zero_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_0*.csv')
one_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_1*.csv')
two_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_2*.csv')
three_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_3*.csv')
four_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_4*.csv')
five_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_5*.csv')
six_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_6*.csv')
seven_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_7*.csv')
eight_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_8*.csv')
nine_subjects = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/sub_9*.csv')

#'''
'''
zero_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_0*.csv')
one_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_1*.csv')
two_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_2*.csv')
three_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_3*.csv')
four_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_4*.csv')
five_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_5*.csv')
six_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_6*.csv')
seven_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_7*.csv')
eight_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_8*.csv')
nine_subjects = glob.glob('size_30sec_150ts_stride_03ts/sub_9*.csv')

'''
print("Generalization Testing: ")

print("Begin iterative generalization testing: ")


for split in range(1, 5):
    train_subjects = 0
    x_train = pd.DataFrame()
    for i in range(split*2, 10):
        if(i == 1):
            for csv_file in one_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1
        elif(i == 2):
            for csv_file in two_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 3):
            for csv_file in three_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 4):
            for csv_file in four_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 5):
            for csv_file in five_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 6):
            for csv_file in six_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 7):
            for csv_file in seven_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 8):
            for csv_file in eight_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

        elif(i == 9):
            for csv_file in nine_subjects:
                df = pd.read_csv(csv_file)
                x_train = pd.concat([x_train, df])
                train_subjects = train_subjects + 1

    print("# Of Train Subjects: " + str(train_subjects))

    test_subjects = 0
    x_test = pd.DataFrame()
    for i in range(0, split*2):
        if(i == 0):
            for csv_file in zero_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1
        elif(i == 1):
            for csv_file in one_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 2):
            for csv_file in two_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 3):
            for csv_file in three_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 4):
            for csv_file in four_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 5):
            for csv_file in five_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 6):
            for csv_file in six_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1

        elif(i == 7):
            for csv_file in seven_subjects:
                df = pd.read_csv(csv_file)
                x_test = pd.concat([x_test, df])
                test_subjects = test_subjects + 1



    print("# Of Test Subjects: " + str(test_subjects))





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


    x_train = x_train.astype(np.float32)

    y_train = np.array(y_train)
    y_train = y_train.reshape(reshape, 150, 2)
    array = np.zeros(reshape, )
    for i in range(0, reshape - 1):
        array[i] = y_train[i][0][1]


    y_train = array
    y_train = y_train.astype(np.int8)

    input = tf.keras.layers.Input(shape = (150, 8))


    y_test = x_test.loc[:, ['chunk', 'label']]
    x_test.pop('chunk')
    x_test.pop('label')


    x_test = np.array(x_test)
    print(x_test.shape)
    reshape = int(x_test.shape[0]/150)
    print(reshape)
    x_test = x_test.reshape(reshape, 150, 8)


    x_test = x_test.astype(np.float32)

    y_test = np.array(y_test)
    y_test = y_test.reshape(reshape, 150, 2)
    array = np.zeros(reshape, )
    for i in range(0, reshape - 1):
        array[i] = y_test[i][0][1]


    y_test = array
    y_test = y_test.astype(np.int8)



    base_lr = .02
    train_steps = reshape // 64
    decay_lr = .66
    clipnorm = .9999


    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            base_lr, train_steps, decay_lr
        )

    LTC_NCP_model = LTC_NCP(input, 100, 5, .5)
    LTC_FullyConnected_model = LTC_FullyConnected(input, 100, 5, .2)
    CNN_model = CNN(input)



    ncp_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

    ncp_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    ltc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

    ltc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    cnn_optimizer = tf.keras.optimizers.Adam()

    cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    CNN_model.compile(optimizer = cnn_optimizer, loss = cnn_loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())
    LTC_NCP_model.compile(ncp_optimizer, ncp_loss,  metrics = tf.keras.metrics.SparseCategoricalAccuracy())
    LTC_FullyConnected_model.compile(ltc_optimizer, ltc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    LTC_NCP_model.fit(x_train, y_train, batch_size = 64, epochs = 20)
    LTC_FullyConnected_model.fit(x_train, y_train, batch_size = 64, epochs = 20)
    CNN_model.fit(x_train, y_train, batch_size = 64, epochs = 17)
    
    CNN_results = CNN_model.evaluate(x_test, y_test, verbose = 1)
    LTC_NCP_results = LTC_NCP_model.evaluate(x_test, y_test, verbose = 1)
    LTC_FC_results = LTC_FullyConnected_model.evaluate(x_test, y_test, verbose = 1)

    
    LTC_NCP_accuracy.append(LTC_NCP_results[1])
    LTC_FC_accuracy.append(LTC_FC_results[1])
    CNN_accuracy.append(CNN_results[1])


'''

for i in range(1, 80, 5):
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = float((i)/100), shuffle = True)
    LTC_NCP_model = LTC_NCP(input, 100, 5, .5)
    LTC_FullyConnected_model = LTC_FullyConnected(input, 100, 5, .2)
    CNN_model = CNN(input)



    ncp_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

    ncp_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    ltc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

    ltc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    cnn_optimizer = tf.keras.optimizers.Adam()

    cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    CNN_model.compile(optimizer = cnn_optimizer, loss = cnn_loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())
    LTC_NCP_model.compile(ncp_optimizer, ncp_loss,  metrics = tf.keras.metrics.SparseCategoricalAccuracy())
    LTC_FullyConnected_model.compile(ltc_optimizer, ltc_loss, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    LTC_NCP_model.fit(x_train, y_train, batch_size = 64, epochs = 20)
    LTC_FullyConnected_model.fit(x_train, y_train, batch_size = 64, epochs = 20)
    CNN_model.fit(x_train, y_train, batch_size = 64, epochs = 17)
    
    CNN_results = CNN_model.evaluate(x_valid, y_valid, verbose = 1)
    LTC_NCP_results = LTC_NCP_model.evaluate(x_valid, y_valid, verbose = 1)
    LTC_FC_results = LTC_FullyConnected_model.evaluate(x_valid, y_valid, verbose = 1)
    #print("Noise: " + str(float(float(i)/100)))
    #print("CNN_accuracy: " + str(CNN_results[1]))
    #print("LTC_NCP accuracy: " + str(LTC_NCP_results[1]))
    split_x.append(float(float(i) / 100))
    CNN_accuracy.append(CNN_results[1])
    LTC_NCP_accuracy.append(LTC_NCP_results[1])
    LTC_FC_accuracy.append(LTC_FC_results[1])

    
'''


'''

'''
#plt.show(block = True)
print("Amount of training subjects: ")
for i in split_x:
    print(i)

print("CNN accuracy: ")
for i in CNN_accuracy:
    print(i)


print("LTC_NCP accuracy")
for i in LTC_NCP_accuracy:
    print(i)

print("LTC_FC_accuracy")
for i in LTC_FC_accuracy:
    print(i)

print("Finished, did GENERALIZATION")

