#!/usr/bin/env python3
""" regularize"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """create l2 regularize layer with tf"""
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=weight, kernel_regularizer=reg)
    return L(prev)
