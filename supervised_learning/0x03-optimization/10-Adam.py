#!/usr/bin/env python3
""" mini batch"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ update with adam tensorflow"""
    a = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return a.minimize(loss)
