#!/usr/bin/env python3
""" convolutions"""
import tensorflow.keras as K


def lenet5(X):
    """ lenet with keras"""
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal")(X)
    pool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                            activation='relu', padding="valid",
                            kernel_initializer="he_normal")(pool1)
    pool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2_flat = K.layers.Flatten()(pool2)
    FL1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer="he_normal")(pool2_flat)
    FL2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer="he_normal")(FL1)
    FL3 = K.layers.Dense(units=10, activation='softmax',
                         kernel_initializer="he_normal")(FL2)
    opt = K.optimizers.Adam()
    model = K.Model(inputs=X, outputs=FL3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=["accuracy"])
    return model
