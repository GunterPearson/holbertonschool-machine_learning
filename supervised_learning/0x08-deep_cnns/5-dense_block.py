#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ dense block"""
    for x in range(layers):
        batch = K.layers.BatchNormalization(axis=3)(X)
        act = K.layers.Activation('relu')(batch)
        conv = K.layers.Conv2D(filters=128,
                               strides=1,
                               kernel_size=(1, 1),
                               kernel_initializer='he_normal',
                               padding='same')(act)
        batch1 = K.layers.BatchNormalization(axis=3)(conv)
        act1 = K.layers.Activation('relu')(batch1)
        conv1 = K.layers.Conv2D(filters=growth_rate,
                                strides=1,
                                kernel_size=(3, 3),
                                kernel_initializer='he_normal',
                                padding='same')(act1)
        concat = K.layers.Concatenate(axis=3)([X, conv1])
        X = concat
    return concat, concat._shape_val[-1]
