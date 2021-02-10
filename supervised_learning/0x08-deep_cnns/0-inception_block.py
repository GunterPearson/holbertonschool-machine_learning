#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception"""
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filters[4]
    FPP = filters[5]
    conv1 = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            padding='same',
                            activation='relu')(A_prev)
    conv3a = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(A_prev)
    conv3b = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu')(conv3a)
    conv5a = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu')(A_prev)
    conv5b = K.layers.Conv2D(filters=F5,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu')(conv5a)
    pool = K.layers.MaxPool2D(pool_size=(3, 3),
                              strides=(1, 1),
                              padding='same')(A_prev)
    convp = K.layers.Conv2D(filters=FPP,
                            kernel_size=(1, 1),
                            padding='same',
                            activation='relu')(pool)
    concat = K.layers.Concatenate(axis=3)([conv1, conv3b, conv5b, convp])
    return concat
