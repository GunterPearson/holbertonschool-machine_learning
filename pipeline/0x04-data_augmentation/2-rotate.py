#!/usr/bin/env python3
"""rotate image"""
import tensorflow as tf


def rotate_image(image):
    """that rotates an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image)
