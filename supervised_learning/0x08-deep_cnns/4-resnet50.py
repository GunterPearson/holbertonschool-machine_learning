#!/usr/bin/env python3
""" inception"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """the keras model"""
    inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(2, 2))(inputs)
    BN1 = K.layers.BatchNormalization(axis=3)(conv1)
    Relu1 = K.layers.Activation("relu")(BN1)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding="same")(Relu1)
    PB1 = projection_block(pool1, [64, 64, 256], s=1)
    ID1 = identity_block(PB1, [64, 64, 256])
    ID2 = identity_block(ID1, [64, 64, 256])
    PB2 = projection_block(ID2, [128, 128, 512])
    ID3 = identity_block(PB2, [128, 128, 512])
    ID4 = identity_block(ID3, [128, 128, 512])
    ID5 = identity_block(ID4, [128, 128, 512])
    PB3 = projection_block(ID5, [256, 256, 1024])
    ID6 = identity_block(PB3, [256, 256, 1024])
    ID7 = identity_block(ID6, [256, 256, 1024])
    ID8 = identity_block(ID7, [256, 256, 1024])
    ID9 = identity_block(ID8, [256, 256, 1024])
    ID10 = identity_block(ID9, [256, 256, 1024])
    PB4 = projection_block(ID10, [512, 512, 2048])
    ID11 = identity_block(PB4, [512, 512, 2048])
    ID12 = identity_block(ID11, [512, 512, 2048])
    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(ID12)
    lin = K.layers.Dense(units=1000, activation='softmax',
                         kernel_initializer="he_normal")(pool2)
    return K.Model(inputs=inputs, outputs=lin)
