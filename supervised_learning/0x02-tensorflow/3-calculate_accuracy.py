#!/usr/bin/env python3
""" create layer """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculate accuracy"""
    p_m = tf.arg_max(y_pred, 1)
    y_m = tf.arg_max(y, 1)
    e = tf.equal(y_m, p_m)
    return tf.reduce_mean(tf.cast(e, tf.float32))
