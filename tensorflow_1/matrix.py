import os
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES "] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


x = tf.constant([[10., 20.]])
y = tf.constant([[30.], [40.]])

z = tf.matmul(x, y)

with tf.Session(config=config) as sess:
    print(sess.run(z))