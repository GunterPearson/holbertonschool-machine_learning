#!/usr/bin/env python3
""" regularize"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ create dropout layer with tf"""
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(1 - keep_prob)
    L = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=weight, kernel_regularizer=drop)
    return L(prev)
