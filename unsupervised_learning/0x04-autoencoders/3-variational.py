#!/usr/bin/env python3
"""Auto Encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    def sampling(inputs):
        mean, log_var = inputs
        tf_shape = keras.backend.shape(log_var)
        norm = keras.backend.random_normal(tf_shape)
        return norm * keras.backend.exp(log_var / 2) + mean

    input = keras.Input(shape=(input_dims,))
    encode = keras.layers.Dense(hidden_layers[0], activation='relu')(input)
    encode_mean = keras.layers.Dense(latent_dims)(encode)
    encode_log = keras.layers.Dense(latent_dims)(encode)
    encoding = keras.layers.Lambda(sampling)([encode_mean, encode_log])
    encoder = keras.Model(input, [encode_mean, encode_log, encoding])

    input2 = keras.Input(shape=(latent_dims,))
    decode = keras.layers.Dense(hidden_layers[-1], activation='relu')(input2)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(input2, decode)

    auto = keras.Model(input, decoder(encoder(input)[-1]))

    k_exp = keras.backend.exp(encode_log)
    k_square = keras.backend.square(encode_mean)
    lat_loss = -0.5 * keras.backend.sum(1 + encode_log - k_exp
                                        - k_square, axis=-1)

    auto.add_loss(keras.backend.mean(lat_loss / 784.))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
