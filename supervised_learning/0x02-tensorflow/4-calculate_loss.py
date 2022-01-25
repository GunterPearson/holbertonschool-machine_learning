#!/usr/bin/env python3
""" calcualte loss"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ loss function"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
