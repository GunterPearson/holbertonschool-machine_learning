#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ transition layer"""
    batch = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(batch)
    conv = K.layers.Conv2D(filters=int(nb_filters * compression),
                           strides=1,
                           kernel_size=(1, 1),
                           kernel_initializer='he_normal',
                           padding='same')(act)
    avpool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(conv)
    return avpool, avpool._shape_val[-1]
