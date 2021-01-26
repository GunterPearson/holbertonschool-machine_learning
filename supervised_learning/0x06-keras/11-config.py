#!/usr/bin/env python3
""" train """
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model’s configuration in JSON format"""
    json = network.to_json()
    with open(filename, 'w') as f:
        f.write(json)
    return None


def load_config(filename):
    """loads a model with a specific configuration:"""
    with open(filename, 'r') as f:
        json = f.read()
    return K.models.model_from_json(json)
