import os
import tensorflow.compat.v1 as tf

#选择所用的gpu 0 or 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

x = tf.constant([[10., 20.]])
y = tf.constant([[30.], [40.]])

#乘法矩阵
z = tf.matmul(x, y)

with tf.Session(config=config) as sess:
    print(sess.run(z))

