#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ resent model"""
    input = K.Input(shape=(224, 224, 3))
    conv = K.layers.Conv2D(filters=64,
                           strides=2,
                           kernel_size=(7, 7),
                           kernel_initializer='he_normal',
                           padding='same')(input)
    batch = K.layers.BatchNormalization(axis=3)(conv)
    act = K.layers.Activation('relu')(batch)
    pool = K.layers.MaxPool2D(pool_size=(3, 3),
                              strides=2,
                              padding='same')(act)
    proj = projection_block(pool, [64, 64, 256], s=1)
    id = identity_block(proj, [64, 64, 256])
    id1 = identity_block(id, [64, 64, 256])
    proj1 = projection_block(id1, [128, 128, 512])
    id2 = identity_block(proj1, [128, 128, 512])
    id3 = identity_block(id2, [128, 128, 512])
    id4 = identity_block(id3, [128, 128, 512])
    proj2 = projection_block(id4, [256, 256, 1024])
    id5 = identity_block(proj2, [256, 256, 1024])
    id6 = identity_block(id5, [256, 256, 1024])
    id7 = identity_block(id6, [256, 256, 1024])
    id8 = identity_block(id7, [256, 256, 1024])
    id9 = identity_block(id8, [256, 256, 1024])
    proj3 = projection_block(id9, [512, 512, 2048])
    id10 = identity_block(proj3, [512, 512, 2048])
    id11 = identity_block(id10, [512, 512, 2048])
    avpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       strides=(1, 1))(id11)
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer='he_normal')(avpool)
    model = K.models.Model(inputs=input, outputs=dense)
    return model
