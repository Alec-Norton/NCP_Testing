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
import keras_tuner as kt

#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')


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



input = tf.keras.layers.Input(shape = (150, 8))

def LTC_FullyConnected_model_builder(hp):

    units = hp.Int('units', min_value = 10, max_value = 100, step = 1)
    

    wiring = ncps.wirings.FullyConnected(units)

    mixed_memory = hp.Boolean('mixed_memory', default = False)

    #backbone_units = hp.Int('backbone_units', min_value = 64, max_value = 256, step = 32)
    #backbone_layers = hp.Int('backbone_layer', min_value = 0, max_value = 3, step = 1)
    #backbone_dropout = hp.Float('backbone_dropout', min_value = 0, max_value = .9, step = .1)

    

    x = LTC(wiring, mixed_memory = mixed_memory, return_sequences= True)(input)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)

    hp_learning_rate = hp.Choice('learning_rate', values = [.001, .005, .01, .015, .02])
    hp_clipnorm = hp.Float('clipnorm', min_value = .25, max_value = 5, step = .25)
    train_steps = reshape // 32
    decay_lr = hp.Float('decay rate', min_value = .5, max_value = .95, step = .5)



    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        hp_learning_rate, train_steps, decay_lr
    )

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate, clipnorm = hp_clipnorm),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

tuner = kt.Hyperband(LTC_FullyConnected_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     overwrite = True,
                     directory = '/home/arnorton/NCP_Testing',
                     project_name = "LTC_Fully_Connected")

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode = "min", patience = 5)
stop_early1 = tf.keras.callbacks.TerminateOnNaN()

tuner.search(x_train, y_train, epochs = 50, validation_data = (x_valid, y_valid), callbacks = [stop_early, stop_early1])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. Optimal values below: 
      units = {best_hps.get('units')},
      mixed memory = {best_hps.get('mixed_memory')},
      backbone_activation = {best_hps.get('backbone_activation')},
      learning_rate = {best_hps.get('learning_rate')},
      decay_rate = {best_hps.get('decay rate')},
      clipnorm = {best_hps.get('clipnorm')}

""")
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_valid, y_valid))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.summary()



# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_data = (x_valid, y_valid))

eval_result = hypermodel.evaluate(x_valid, y_valid)
print("[test loss, test accuracy]:", eval_result)
