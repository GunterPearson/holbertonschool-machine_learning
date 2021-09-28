#!/usr/bin/env python3
"""RNN Decoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class"""
    def __init__(self, vocab, embedding, units, batch) -> None:
        """class Initializer"""
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Instance call"""
        self_attention = SelfAttention(self.units)
        context, _ = self_attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        inputs = tf.concat([context, x], axis=-1)
        output, state = self.gru(inputs=inputs)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, state
