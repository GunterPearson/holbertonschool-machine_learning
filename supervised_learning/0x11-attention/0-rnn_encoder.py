#!/usr/bin/env python3
"""RNN Encoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """enocde layer inherits from keras layer"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """Initializes hidden states RNN cell to a tensor of zeros"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Returns: outputs, hidden"""
        embedded = self.embedding(x)
        return self.gru(inputs=embedded, initial_state=initial)
