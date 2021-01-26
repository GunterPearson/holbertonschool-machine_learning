#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ train model using mini-batch"""
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
