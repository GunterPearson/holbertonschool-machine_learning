#!/usr/bin/env python3
"""Auto Encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder"""
    # autoencoder(784, [128, 64], 32)
    encode = keras.Sequential()
    encode.add(keras.layers.Dense(hidden_layers[0],
                                  activation='relu', input_dim=input_dims))
    for lay in hidden_layers[1:]:
        encode.add(keras.layers.Dense(lay, activation='relu'))
    encode.add(keras.layers.Dense(latent_dims, activation='relu'))

    decode = keras.Sequential()
    decode.add(keras.layers.Dense(hidden_layers[1],
                                  activation='relu', input_dim=latent_dims))
    for lay in hidden_layers[-2::-1]:
        decode.add(keras.layers.Dense(lay, activation='relu'))
    decode.add(keras.layers.Dense(input_dims, activation='sigmoid'))

    auto = keras.Sequential()
    auto.add(encode)
    auto.add(decode)

    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encode, decode, auto
