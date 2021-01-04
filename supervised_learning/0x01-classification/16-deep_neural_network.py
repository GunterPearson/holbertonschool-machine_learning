#!/usr/bin/env python3
""" Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """ creating deepNN class"""

    def __init__(self, nx, layers):
        """ initialize deepNN"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        for x in range(self.L):
            if layers[x] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if x == 0:
                self.weights = {"W1": np.random.randn(layers[0],
                                nx) * np.sqrt(2 / nx),
                                "b1": np.zeros((layers[0], 1))}
            else:
                W = "W" + str(x + 1)
                B = "b" + str(x + 1)
                self.weights[W] = np.random.randn(
                                  layers[x],
                                  layers[x - 1]) * np.sqrt(2 / layers[x - 1])
                self.weights[B] = np.zeros((layers[x], 1))
