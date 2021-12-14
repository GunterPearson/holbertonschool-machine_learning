#!/usr/bin/env python3
"""crop image"""
import tensorflow as tf


def crop_image(image, size):
    """that performs a random crop of an image"""
    cropped_image = tf.image.random_crop(image, size)
    return cropped_image
