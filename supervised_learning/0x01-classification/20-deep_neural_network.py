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
        self.__L = len(layers)
        self.__cache = {}
        for x in range(self.L):
            if layers[x] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if x == 0:
                self.__weights = {"W1": np.random.randn(layers[0],
                                  nx) * np.sqrt(2 / nx),
                                  "b1": np.zeros((layers[0], 1))}
            else:
                W = "W" + str(x + 1)
                B = "b" + str(x + 1)
                self.__weights[W] = np.random.randn(
                                  layers[x],
                                  layers[x - 1]) * np.sqrt(2 / layers[x - 1])
                self.__weights[B] = np.zeros((layers[x], 1))

    @property
    def L(self):
        """ return private w"""
        return self.__L

    @property
    def cache(self):
        """ return private b"""
        return self.__cache

    @property
    def weights(self):
        """ return private a"""
        return self.__weights

    def forward_prop(self, X):
        """ forward prop for deep neural network"""
        self.__cache["A0"] = X
        for x in range(self.__L):
            n = str(x + 1)
            Z = np.matmul(
                self.__weights["W" + n],
                self.__cache["A" + str(x)]) + self.__weights["b" + n]
            self.__cache["A" + n] = 1/(1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ return the cost """
        cost = -(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))).mean()
        return cost

    def evaluate(self, X, Y):
        """ evaluate to binary 1 or 0"""
        self.forward_prop(X)
        return np.round(self.__cache["A" + str(self.__L)]).astype(
            int), self.cost(Y, self.__cache["A" + str(self.__L)])
