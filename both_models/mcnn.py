#Code provided by https://github.com/max-kuk/MorphConvHyperNet
#MIT licensed
#This file creates the compiled MorphConvHyperNet model, with custom layers from layers.py

#No major edits have been made to this file, with exception of attempting to use jit compilation for non custom layers 

import layers
import tensorflow as tf
tf.config.optimizer.set_jit(True)

print(tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))



def get_compiled_model(shapeinput, num_class):
    # As in paper B is a number of channels
    B = int(shapeinput[-1])
    input_layer = tf.keras.layers.Input(shapeinput)

    x = tf.keras.layers.Conv2D(B // 4, kernel_size=1, padding='same')(input_layer)
    out1 = layers.SpectralMorph(num_filters=int(B / 4))(x)
    out2 = layers.SpatialMorph(num_filters=int(B / 4))(x)
    x = tf.keras.layers.Add()([out1, out2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(int(B / 2), kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(units=num_class, activation='softmax')(x)
    clf = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    clf.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1), optimizer='adam', metrics=['accuracy'], jit_compile = False)
    return clf
