#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ make a prediction"""
    return network.predict(data, verbose=verbose)
