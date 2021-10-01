#!/usr/bin/env python3
"""Positional Encoding"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer"""
    def get_angles(pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dm))
        return pos * angle_rates

    position = np.arange(max_seq_len)
    pos_emb = get_angles(position[:, np.newaxis], np.arange(dm)[np.newaxis, :])
    pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])
    pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])
    return pos_emb


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


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class"""
    def __init__(self, dm, h) -> None:
        """class constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def split_heads(self, x, batch):
        """Splits inputs"""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Instance call"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)
        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder Block class"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """transformer call"""
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layernorm1(x + attention_output)
        dense_output = self.dense_hidden(output1)
        f_output = self.dense_output(dense_output)
        f_output = self.dropout2(f_output, training=training)
        output2 = self.layernorm2(output1 + f_output)
        return output2


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder Block class"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Decoder call"""
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        output1 = self.layernorm1(x + attention_output1)
        attention_output2, _ = self.mha2(output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        output2 = self.layernorm2(output1 + attention_output2)
        dense_output = self.dense_hidden(output2)
        f_output = self.dense_output(dense_output)
        f_output = self.dropout3(f_output, training=training)
        output3 = self.layernorm3(output2 + f_output)
        return output3


class Encoder(tf.keras.layers.Layer):
    """Encoder class"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Class constructor"""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Encoder call"""
        input_seq_len = x.shape[1]
        embedded = self.embedding(x)
        s = embedded * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positioned = s + self.positional_encoding[:input_seq_len, :]
        x = self.dropout(positioned, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder class"""
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """class constructor"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """decoder call"""
        input_seq_len = x.shape[1]
        embedded = self.embedding(x)
        s = embedded * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positioned = s + self.positional_encoding[:input_seq_len, :]
        x = self.dropout(positioned, training=training)
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.layers.Layer):
    """Class to create the transformer network"""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Class constructor"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Calls the transformer"""
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        result = self.linear(decoder_output)
        return result
