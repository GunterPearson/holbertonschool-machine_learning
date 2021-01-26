#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def save_model(network, filename):
    """saves a model"""
    network.save(filename)
    return None


def load_model(filename):
    """loads a model"""
    return K.models.load_model(filename)
