#!/usr/bin/env python3
"""Auto Encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder"""
    # autoencoder(784, [128, 64], 32)
    input = keras.Input(shape=(input_dims,))
    encoded1 = keras.layers.Dense(hidden_layers[0], activation='relu')(input)
    for lay in hidden_layers[1:]:
        encoded2 = keras.layers.Dense(lay, activation='relu')(encoded1)
    encoded3 = keras.layers.Dense(latent_dims, activation='relu')(encoded2)
    encoder = keras.Model(input, encoded3)

    input2 = keras.Input(shape=(latent_dims,))
    decoded1 = keras.layers.Dense(hidden_layers[-1], activation='relu')(input2)
    for dim in hidden_layers[-2::-1]:
        decoded2 = keras.layers.Dense(dim, activation='relu')(decoded1)
    decoded3 = keras.layers.Dense(input_dims, activation='sigmoid')(decoded2)
    decoder = keras.Model(input2, decoded3)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
