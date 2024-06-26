import sys
import argparse




parser = argparse.ArgumentParser()

#For Wiring
parser.add_argument("size")

#For opt
parser.add_argument("base_lr")
parser.add_argument("clipnorm")
#CfC args
parser.add_argument("batch_size")
parser.add_argument("epochs")
parser.add_argument("model_number")

args = parser.parse_args()

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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



keras = tf.keras
#define a function to return a NCP CfC Model
def CFC_FullyConnected(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.FullyConnected(ncp_size)
    #Begin constructing layer, starting with input

    x = CfC(wiring, return_sequences= True)(input)
    x = keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model


def eval(model, index_arg, train_x, train_y, x_valid, y_valid, opt, loss_fun, batch_size, epochs):
    #Compile the Model
    model.compile(optimizer = opt, loss = loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    #Return a summary of the model and its (non)trainable paramaters
    model.summary()

    #Fit the model and return accuracy
    #Get beginning of time
    callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

    start = time.process_time()
    hist = model.fit(x = train_x, y = train_y, validation_data = (x_valid, y_valid), batch_size = batch_size, epochs = epochs, verbose = 2, callbacks = [callback])
    end = time.process_time()
    test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    print("Max Accuracy Of Model: " + str(np.max(test_accuracies)))
    return np.max(test_accuracies), end-start

#Based on the model_number, create a model and train on specified optimizer, loss_function, validation_split, batch_size, and some epochs
#Then return the mean and standard deviation of the accuracy of these models. 
def score(model, train_x, train_y, x_valid, y_valid, opt, loss_fun, model_number, batch_size, epochs):
    acc = []
    dur = []
    for i in range(model_number):
        print("Model: " + str(i))
        max_accuracy, time = eval(model, i, train_x, train_y, x_valid, y_valid, opt, loss_fun, batch_size, epochs)
        dur.append(time)
        acc.append(100 * max_accuracy)
    acc_average = np.mean(acc)
    acc_std = np.std(acc)
    dur_average = np.mean(dur)
    dur_std = np.std(dur)
    print("-------------------------------------------------------------------")
    print("Average Test Accuracy: " + str(acc_average) + " Standard Deviation Test Accuracy: " + str(acc_std))
    print("Average Time Training: " + str(dur_average) + " Standard Deviation Time: " + str(dur_std))
    print("-------------------------------------------------------------------")


    #print(f"Test Accuracy: {np.mean(acc):(1/model_number)}\\% $\\pm$ {np.std(acc):(1/model_number)}")


#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')


x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])



'''
csv_file1 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_97.csv')
csv_file2 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_1.csv')
csv_file3 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_15.csv')
csv_file4 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_31.csv')
csv_file5 = pd.read_csv('size_30sec_150ts_stride_03ts\sub_45.csv')

x_train = csv_file1.copy()
x_train = pd.concat([x_train, csv_file2])
x_train = pd.concat([x_train, csv_file3])
x_train = pd.concat([x_train, csv_file4])
x_train = pd.concat([x_train, csv_file5])
'''

y_train = x_train.iloc[:, [8, 9]]
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



input = tf.keras.layers.Input(shape = (150, 8))

#CfC NCP
ncp_size = int(args.size)


number_of_models = int(args.model_number)
batch_size = int(args.batch_size)
epochs = int(args.epochs)

base_lr = float(args.base_lr)
train_steps = reshape // batch_size
decay_lr = .95
clipnorm = float(args.clipnorm)



learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )


cfc_optimizer = tf.keras.optimizers.Adam(learning_rate_fn ,clipnorm = clipnorm)

cfc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)





score(CFC_FullyConnected(input, ncp_size), x_train, y_train, x_valid, y_valid, cfc_optimizer, cfc_loss, number_of_models, batch_size, epochs)

print("\n")
print("base_lr = " + str(base_lr) + " decay_lr = " + str(decay_lr) + " clipnorm = " + str(clipnorm))
print("\n")
print("Size of Model: " + str(ncp_size))
print("\n")
print("Epochs: " + str(epochs) + " Batch Size: " + str(batch_size) + " Number Of Models: " + str(number_of_models))