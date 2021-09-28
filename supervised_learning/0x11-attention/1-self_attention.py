#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Calculate attention for machine translation"""
    def __init__(self, units):
        """class constructor"""
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Instance Call"""
        s_prev_time = tf.expand_dims(s_prev, 1)
        out = self.V(
            tf.nn.tanh(
                self.W(s_prev_time) + self.U(hidden_states)
            )
        )
        weights = tf.nn.softmax(out, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
