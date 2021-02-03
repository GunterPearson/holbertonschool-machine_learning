#!/usr/bin/env python3
""" convolutions"""
import tensorflow.keras as K


def lenet5(X):
    """ lenet with keras"""
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(X)
    M1 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C1)
    C2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(M1)
    M2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C2)
    CF = K.layers.Flatten()(C2)
    F1 = K.layers.Dense(units=84, activation='relu')(CF)
    F2 = K.layers.Dense(units=120, activation='relu')(F1)
    F3 = K.layers.Dense(units=10, activation='softmax')(F2)
    model = K.Model(inputs=X, outputs=F3)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=K.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model
