#!/usr/bin/env python3
""" mini batch"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ update using rms tensorflow"""
    r = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return r.minimize(loss)
