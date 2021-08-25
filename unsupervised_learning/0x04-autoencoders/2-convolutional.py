#!/usr/bin/env python3
"""Auto Encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder"""
    input = keras.Input(shape=input_dims)
    encode = keras.layers.Conv2D(filters[0], kernel_size=(3, 3),
                                 padding='same', activation='relu')(input)
    encode = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(encode)
    for fil in filters[1:]:
        encode = keras.layers.Conv2D(fil, kernel_size=(3, 3),
                                     padding='same', activation='relu')(encode)
        encode = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        padding='same')(encode)
    encoder = keras.Model(input, encode)

    input2 = keras.Input(shape=latent_dims)
    decode = keras.layers.Conv2D(filters[-1], kernel_size=(3, 3),
                                 padding='same', activation='relu')(input2)
    decode = keras.layers.UpSampling2D(size=(2, 2))(decode)
    decode = keras.layers.Conv2D(filters[-2], kernel_size=(3, 3),
                                 padding='same', activation='relu')(decode)
    decode = keras.layers.UpSampling2D(size=(2, 2))(decode)
    decode = keras.layers.Conv2D(filters[0], kernel_size=(3, 3),
                                 padding='valid', activation='relu')(decode)
    decode = keras.layers.UpSampling2D(size=(2, 2))(decode)
    channel = input_dims[-1]
    decode = keras.layers.Conv2D(channel, kernel_size=(3, 3),
                                 padding='same', activation='sigmoid')(decode)
    decoder = keras.Model(input2, decode)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
