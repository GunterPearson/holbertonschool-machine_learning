#!/usr/bin/env python3
""" batch norm"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ create batch normilization layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=w)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    r = tf.nn.batch_normalization(layer(prev), mean, variance,
                                  beta, gamma, 1e-8)
    return activation(r)
