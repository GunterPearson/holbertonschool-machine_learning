#!/usr/bin/env python3
"""flip image"""
import tensorflow as tf


def flip_image(image):
    """that flips an image horizontally"""
    return tf.image.flip_left_right(image)
