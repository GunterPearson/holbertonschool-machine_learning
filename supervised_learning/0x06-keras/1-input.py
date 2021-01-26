#!/usr/bin/env python3
""" keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model"""
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    new = K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_regularizer=reg)(inputs)
    for x in range(1, len(layers)):
        new = K.layers.Dropout(1 - keep_prob)(new)
        new = K.layers.Dense(layers[x],
                                 activation=activations[x],
                                 kernel_regularizer=reg)(new)
    model = K.Model(inputs=inputs, outputs=new)
    return model
