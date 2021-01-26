#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ train model"""
    call = []
    if early_stopping is True and validation_data is not None:
        early = K.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience,
                                          mode='min')
        call.append(early)
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          callbacks=call,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
