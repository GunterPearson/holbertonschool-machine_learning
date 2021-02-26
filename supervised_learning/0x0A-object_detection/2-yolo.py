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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filter boxes"""
        shape_boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])
        shape_prob = np.concatenate([b.reshape(-1, 80) for b in
                                    box_class_probs])
        shape_conf = np.concatenate([b.reshape(-1) for b in box_confidences])

        final_class = np.argmax(shape_prob, axis=1)
        final_conf = shape_conf * shape_prob.max(axis=1)
        rm = np.where(final_conf < self.class_t)

        shape_boxes = np.delete(shape_boxes, rm, axis=0)
        final_conf = np.delete(final_conf, rm)
        final_class = np.delete(final_class, rm)

        return shape_boxes, final_class, final_conf
