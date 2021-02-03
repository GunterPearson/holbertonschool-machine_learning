#!/usr/bin/env python3
""" convolutions"""
import tensorflow as tf


def lenet5(x, y):
    """ lenet"""
    W = tf.contrib.layers.variance_scaling_initializer()
    c1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding="same",
                          kernel_initializer=W, activation="relu")(x)
    p1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(c1)
    c2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding="valid",
                          kernel_initializer=W, activation="relu")(p1)
    p2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(c2)
    c_prev = tf.layers.Flatten()(p2)
    L1 = tf.layers.Dense(units=120, activation="relu",
                         kernel_initializer=W)(c_prev)
    L2 = tf.layers.Dense(units=84, activation="relu",
                         kernel_initializer=W)(L1)
    y_pred = tf.layers.Dense(units=10, kernel_initializer=W)(L2)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    p_m = tf.arg_max(y_pred, 1)
    y_m = tf.arg_max(y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_m, p_m), tf.float32))
    a = tf.train.AdamOptimizer()
    adam = a.minimize(loss)
    return tf.nn.softmax(y_pred), adam, loss, acc
