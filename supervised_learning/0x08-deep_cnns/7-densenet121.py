#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ densenet"""
    input = K.Input(shape=(224, 224, 3))
    batch = K.layers.BatchNormalization()(input)
    act = K.layers.Activation('relu')(batch)
    conv = K.layers.Conv2D(filters=64,
                           strides=2,
                           kernel_size=(7, 7),
                           kernel_initializer='he_normal',
                           padding='same')(act)
    pool = K.layers.MaxPool2D(pool_size=(3, 3),
                              strides=2,
                              padding='same')(conv)
    db, sh = dense_block(pool, 64, growth_rate, 6)
    tl, sh = transition_layer(db, int(sh), compression)
    db1, sh = dense_block(tl, int(sh), growth_rate, 12)
    tl1, sh = transition_layer(db1, int(sh), compression)
    db2, sh = dense_block(tl1, int(sh), growth_rate, 24)
    tl2, sh = transition_layer(db2, int(sh), compression)
    db3, sh = dense_block(tl2, int(sh), growth_rate, 16)
    avpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       strides=(1, 1))(db3)
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer='he_normal')(avpool)
    model = K.models.Model(inputs=input, outputs=dense)
    return model
