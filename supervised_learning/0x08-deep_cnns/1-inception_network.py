#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ inception_network"""
    input = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64,
                            strides=2,
                            kernel_size=(7, 7),
                            padding='same',
                            activation='relu')(input)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=2,
                               padding='same')(conv1)
    conv3r = K.layers.Conv2D(filters=64,
                             kernel_size=(1, 1),
                             padding="valid",
                             activation='relu')(pool1)
    conv3 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu')(conv3r)
    pool2 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=2,
                               padding='same')(conv3)
    cat1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    cat2 = inception_block(cat1, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=2,
                               padding='same')(cat2)
    cat3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    cat4 = inception_block(cat3, [160, 112, 224, 24, 64, 64])
    cat5 = inception_block(cat4, [128, 128, 256, 24, 64, 64])
    cat6 = inception_block(cat5, [112, 144, 288, 32, 64, 64])
    cat7 = inception_block(cat6, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=2,
                               padding='same')(cat7)
    cat8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    cat9 = inception_block(cat8, [384, 192, 384, 48, 128, 128])
    avpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       strides=1,
                                       padding='valid')(cat9)
    drop = K.layers.Dropout(.3)(avpool)
    dense = K.layers.Dense(1000, activation='softmax')(drop)
    model = K.models.Model(inputs=input, outputs=dense)
    return model
