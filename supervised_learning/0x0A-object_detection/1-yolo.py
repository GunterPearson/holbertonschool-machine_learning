#!/usr/bin/env python3
""" yolo class"""
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
        boxes = [output[:, :, :, 0:4] for output in outputs]
        for oidx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    c_y = ((self.sigmoid(output[y, x, :, 1]) + y)
                           / output.shape[0] * image_size[0])
                    c_x = ((self.sigmoid(output[y, x, :, 0]) + x)
                           / output.shape[1] * image_size[1])
                    resize = self.anchors[oidx].astype(float)
                    resize[:, 0] *= (np.exp(output[y, x, :, 2])
                                     / 2 * image_size[1] /
                                     self.model.input.shape[1].value)
                    resize[:, 1] *= (np.exp(output[y, x, :, 3])
                                     / 2 * image_size[0] /
                                     self.model.input.shape[2].value)
                    output[y, x, :, 0] = c_x - resize[:, 0]
                    output[y, x, :, 1] = c_y - resize[:, 1]
                    output[y, x, :, 2] = c_x + resize[:, 0]
                    output[y, x, :, 3] = c_y + resize[:, 1]
        for output in outputs:
            box_conf.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class.append(self.sigmoid(output[..., 5:]))
        return boxes, box_conf, box_class
