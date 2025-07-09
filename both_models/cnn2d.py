#Code provided by https://github.com/mhaut/hyperspectral_deeplearning_review
#GPL-3.0 licensed
#This file creates the compiled CNN2D model, with standard keras layers

#Adapted to use tensorflow keras, as the MorphConvHyperNet makes use of that version, to keep it consistent

import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

tf.config.optimizer.set_jit(True)

def get_model_compiled(shapeinput, num_class, w_decay=0):
    clf = Sequential()
    clf.add(Conv2D(50, kernel_size=(5, 5), input_shape=shapeinput))
    clf.add(Activation('relu'))
    clf.add(Conv2D(100, (5, 5)))
    clf.add(Activation('relu'))
    clf.add(MaxPooling2D(pool_size=(2, 2)))
    clf.add(Flatten())
    clf.add(Dense(100, kernel_regularizer=regularizers.l2(w_decay)))
    clf.add(Activation('relu'))
    clf.add(Dense(num_class, activation='softmax'))
    clf.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'], jit_compile = True)
    return clf
