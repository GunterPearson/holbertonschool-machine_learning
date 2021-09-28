#!/usr/bin/env python3
"""RNN Decoder"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention"""
    q_k = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = q_k / tf.sqrt(dk)
    if mask is not None:
        scaled += mask
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
