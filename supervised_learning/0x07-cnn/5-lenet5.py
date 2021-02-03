#!/usr/bin/env python3
""" convolutions"""
import tensorflow.keras as K


def lenet5(X):
    """ lenet with keras"""
    activation = 'relu'
    kInit = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(filters=6, kernel_size=5,
                              padding='same',
                              activation=activation,
                              kernel_initializer=kInit)(X)

    pool_1 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_1)

    layer_2 = K.layers.Conv2D(filters=16, kernel_size=5,
                              padding='valid',
                              activation=activation,
                              kernel_initializer=kInit)(pool_1)

    pool_2 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_2)

    flatten = K.layers.Flatten()(pool_2)

    layer_3 = K.layers.Dense(120, activation=activation,
                             kernel_initializer=kInit)(flatten)

    layer_4 = K.layers.Dense(84, activation=activation,
                             kernel_initializer=kInit)(layer_3)

    output_layer = K.layers.Dense(10, activation='softmax',
                                  kernel_initializer=kInit)(layer_4)

    model = K.models.Model(X, output_layer)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
