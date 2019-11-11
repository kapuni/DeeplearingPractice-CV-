import os
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K


#选择所用的gpu 0 or 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True