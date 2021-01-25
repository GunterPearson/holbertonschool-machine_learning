#!/usr/bin/env python3
""" keras"""
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model"""
    model = keras.Sequential()
    reg = keras.regularizers.l2(lambtha)
    for x in range(len(layers)):
        if x == 0:
            model.add(keras.layers.Dense(layers[x],
                      activation=activations[x],
                      kernel_regularizer=reg,
                      input_shape=(nx,)))
        else:
            model.add(keras.layers.Dropout(1 - keep_prob))
            model.add(keras.layers.Dense(layers[x],
                      activation=activations[x],
                      kernel_regularizer=reg))
    return model
