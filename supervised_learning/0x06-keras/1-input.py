#!/usr/bin/env python3
""" keras"""
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model"""
    inputs = keras.Input(shape=(nx,))
    reg = keras.regularizers.l2(lambtha)
    new = keras.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_regularizer=reg)(inputs)
    for x in range(1, len(layers)):
        new = keras.layers.Dropout(1 - keep_prob)(new)
        new = keras.layers.Dense(layers[x],
                                 activation=activations[x],
                                 kernel_regularizer=reg)(new)
    model = keras.Model(inputs=inputs, outputs=new)
    return model
