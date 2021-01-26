#!/usr/bin/env python3
""" keras"""
from tensorflow import keras


def optimize_model(network, alpha, beta1, beta2):
    """ optimize with adam"""
    opt = keras.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy', optimizer=opt)
    return None
