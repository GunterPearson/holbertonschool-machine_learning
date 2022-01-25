#!/usr/bin/env python3
""" train"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ training operation"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
