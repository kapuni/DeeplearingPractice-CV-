import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers as opts

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

K.set_session(tf.Session(config=config))

x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10,
                                                       size=(1000, 1)),
                                     num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10,
                                                       size=(100, 1)),
                                     num_classes=10)
model = keras.models.Sequential()

model.add(layers.Dense(64, activation='relu', input_dim=20))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

sgd = opts.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

print(score)
