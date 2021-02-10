#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Identity_block"""
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(filters=F11,
                            strides=1,
                            kernel_size=(1, 1),
                            kernel_initializer='he_normal',
                            padding='same')(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    conv2 = K.layers.Conv2D(filters=F3,
                            strides=1,
                            kernel_size=(3, 3),
                            kernel_initializer='he_normal',
                            padding='same')(act1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(batch2)
    conv3 = K.layers.Conv2D(filters=F12,
                            strides=1,
                            kernel_size=(1, 1),
                            kernel_initializer='he_normal',
                            padding='same')(act2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)
    add1 = K.layers.Add()([batch3, A_prev])
    act3 = K.layers.Activation('relu')(add1)
    return act3
