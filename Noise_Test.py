import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time 
from sklearn.model_selection import train_test_split
import keras_tuner as kt


import sys

import argparse

keras = tf.keras

CNN_model = keras.models.load_model('CNN_Model.keras')
LTC_NCP_model = keras.models.load_model('LTC_NCP_Model.keras')