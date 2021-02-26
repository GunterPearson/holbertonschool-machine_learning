#!/usr/bin/env python3
""" yolo class"""
from numpy.core.fromnumeric import resize
import tensorflow as tf
import numpy as np


class Yolo():
    """ yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ class constructor"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            line = [s.rstrip('\n') for s in f]
        self.class_names = line
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid function"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """process outputs"""
        boxes = []
        box_conf = []
        box_class = []
        for out in outputs:
            boxes.append(out[..., 0:4])
            box_conf.append(self.sigmoid(out[..., 4, np.newaxis]))
            box_class.append(self.sigmoid(out[..., 5:]))
        return boxes, box_conf, box_class
