#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """ train model"""
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
