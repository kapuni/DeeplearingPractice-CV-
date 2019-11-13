import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers as opts

os.environ["CUDA_VISIBLE_DEVICES "] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

K.set_session(tf.Session(config=config))

a = layers.Input(shape=(32, ))
b = layers.Dense(32)(a)
b = layers.Dropout(0.5)(b)
model = keras.models.Model(input(a), outputs = b)

