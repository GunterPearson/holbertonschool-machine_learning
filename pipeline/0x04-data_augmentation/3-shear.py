#!/usr/bin/env python3
"""shear image"""
import tensorflow as tf


def shear_image(image, intensity):
    """that randomly shears an image"""
    return tf.keras.preprocessing.image.random_shear(image, intensity)
