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

def CfC_NCP_model_builder(hp):

    inter_neuron = hp.Int('inter_neurons', min_value = 5, max_value = 30, step = 1)
    command_neuron = hp.Int('command_neurons', min_value = 5, max_value = 30, step = 1)
    motor_neuron = hp.Int('motor_neurons', min_value = 4, max_value = 30, step = 1)
    sensory_fanout = hp.Int('sensory_fanout', min_value = 1, max_value = int(.9 * inter_neuron), step = 1)
    inter_fanout = hp.Int('inter_fanout', min_value = 1, max_value = int(.9 * command_neuron), step = 1)
    recurrent_command_synapses = hp.Int('recurrent_command_synapses', min_value = 1, max_value = int(1.8 * command_neuron))
    motor_fanin = hp.Int('motor_fanin', min_value = 1, max_value = int(.9 * command_neuron), step = 1)

    wiring = ncps.wirings.NCP(inter_neurons = inter_neuron, command_neurons = command_neuron, motor_neurons = motor_neuron, sensory_fanout = sensory_fanout, inter_fanout = inter_fanout, recurrent_command_synapses= recurrent_command_synapses, motor_fanin= motor_fanin)

    mode = hp.Choice('mode', values = ["default", "pure", "no_gate"])
    backbone_activation = hp.Choice('backbone_activation', values = ["silu", "relu", "tanh", "lecun_tanh", "softplus"])

    #backbone_units = hp.Int('backbone_units', min_value = 64, max_value = 256, step = 32)
    #backbone_layers = hp.Int('backbone_layer', min_value = 0, max_value = 3, step = 1)
    #backbone_dropout = hp.Float('backbone_dropout', min_value = 0, max_value = .9, step = .1)

    

    x = CfC(wiring, mode = mode, activation = backbone_activation, return_sequences= True)(input)
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

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = hp_clipnorm),
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

tuner = kt.Hyperband(CfC_NCP_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     factor = 3,
                     overwrite = True,
                     directory = '',
                     project_name = "CfC_NCP_Tuning_Project")

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode = "min", patience = 5)
stop_early1 = tf.keras.callbacks.TerminateOnNaN()

tuner.search(x_train, y_train, epochs = 50, validation_data = (x_valid, y_valid), callbacks = [stop_early, stop_early1])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data = (x_valid, y_valid))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1


hypermodel = tuner.hypermodel.build(best_hps)

hypermodel.summary()



# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_data = (x_valid, y_valid))

eval_result = hypermodel.evaluate(x_valid, y_valid)
print("CfC_NCP ")
print(f"""
The hyperparameter search is complete. Optimal values below: 
      inter neurons = {best_hps.get('inter_neurons')},
      command neurons = {best_hps.get('command_neurons')},
      motor neurons = {best_hps.get('motor_neurons')},
      sensory fanout = {best_hps.get('sensory_fanout')},
      inter_fanout = {best_hps.get('inter_fanout')},
      recurrent_command_synapses = {best_hps.get('recurrent_command_synapses')},
      motor_fanin = {best_hps.get('motor_fanin')},
      mode = {best_hps.get('mode')},
      backbone_activation = {best_hps.get('backbone_activation')},
      learning_rate = {best_hps.get('learning_rate')},
      clipnorm = {best_hps.get('clipnorm')},
      decay rate = {best_hps.get('decay_rate')}



""")
print('Best epoch: %d' % (best_epoch,))

print("[test loss, test accuracy]:", eval_result)
