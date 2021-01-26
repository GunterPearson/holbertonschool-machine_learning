#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """ train model"""
    call = []
    if early_stopping is True and validation_data is not None:
        early = K.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience,
                                          mode='min')
        call.append(early)
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=scheduler, verbose=1)
        call.append(lr_decay)
    if save_best:
        sa = K.callbacks.ModelCheckpoint(filepath=filepath,
                                         save_best_only=save_best)
        call.append(sa)
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          callbacks=call,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
