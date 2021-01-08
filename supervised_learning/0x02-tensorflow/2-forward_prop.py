#!/usr/bin/env python3
""" create layer """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward prop"""
    layer_output = x
    for i in range(len(layer_sizes)):
        layer_output = create_layer(layer_output, layer_sizes[i],
                                    activations[i])
    return layer_output
