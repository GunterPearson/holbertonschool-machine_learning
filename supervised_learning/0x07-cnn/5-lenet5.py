#!/usr/bin/env python3
""" convolutions"""
import tensorflow.keras as K


def lenet5(X):
    """ lenet with keras"""
    model = K.Sequential()
    L1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                         activation='relu', padding="same",
                         kernel_initializer="he_normal",
                         input_shape=X.shape)(X)
    p1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L1)
    L2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                         activation='relu', padding="valid",
                         kernel_initializer="he_normal")(p1)
    p2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L2)
    flat = K.layers.Flatten()(p2)
    L3 = K.layers.Dense(units=120,
                        activation='relu',
                        kernel_initializer="he_normal")(flat)
    L4 = K.layers.Dense(units=84,
                        activation='relu',
                        kernel_initializer="he_normal")(L3)
    y_pred = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer="he_normal")(L4)
    model = K.Model(inputs=X, outputs=y_pred)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
