#!/usr/bin/env python3
""" keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model"""
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    for x in range(len(layers)):
        if x == 0:
            model.add(K.layers.Dense(layers[x],
                      activation=activations[x],
                      kernel_regularizer=reg,
                      input_shape=(nx,)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[x],
                      activation=activations[x],
                      kernel_regularizer=reg))
    return model
