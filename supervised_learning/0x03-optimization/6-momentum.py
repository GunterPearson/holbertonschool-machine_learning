#!/usr/bin/env python3
""" mini batch"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ update using momentum tensorflow"""
    m = tf.train.MomentumOptimizer(alpha, beta1)
    return m.minimize(loss)
