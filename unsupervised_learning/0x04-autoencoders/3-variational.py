#!/usr/bin/env python3
"""Auto Encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""
    class Sampling(keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            tf_shape = keras.backend.shape(log_var)
            norm = keras.backend.random_normal(tf_shape)
            return norm * keras.backend.exp(log_var / 2) + mean

    input = keras.Input(shape=(input_dims,))
    encode = keras.layers.Dense(hidden_layers[0], activation='relu')(input)
    for dim in hidden_layers[1:]:
        encode = keras.layers.Dense(dim, activation='relu')(encode)
    encode_mean = keras.layers.Dense(latent_dims)(encode)
    encode_log = keras.layers.Dense(latent_dims)(encode)
    encoding = Sampling()([encode_mean, encode_log])
    encoder = keras.Model(input, [encode_mean, encode_log, encoding])

    input2 = keras.Input(shape=(latent_dims,))
    decode = keras.layers.Dense(hidden_layers[-1], activation='relu')(input2)
    for dim in hidden_layers[-2::-1]:
        decode = keras.layers.Dense(dim, activation='relu')(decode)
    decode = keras.layers.Dense(input_dims, activation='sigmoid')(decode)
    decoder = keras.Model(input2, decode)

    auto = keras.Model(input, decoder(encoder(input)[-1]))

    def vae_loss(inputs, outputs):
        r_loss = keras.losses.binary_crossentropy(inputs, outputs)
        r_loss *= input_dims
        k_exp = keras.backend.exp(encode_log)
        k_square = keras.backend.square(encode_mean)
        lat_loss = -0.5 * keras.backend.sum(1 + encode_log - k_exp
                                            - k_square, axis=-1)
        vae = keras.backend.mean(lat_loss + r_loss)
        return vae

    auto.compile(loss=vae_loss, optimizer='adam')

    return encoder, decoder, auto
