#!/usr/bin/env python3
""" neural network class"""
import numpy as np


class NeuralNetwork:
    """ Creating Neural Network class"""

    def __init__(self, nx, nodes):
        """ initialize class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ return private w1"""
        return self.__W1

    @property
    def b1(self):
        """ return private b1"""
        return self.__b1

    @property
    def A1(self):
        """ return private A1"""
        return self.__A1

    @property
    def W2(self):
        """ return private w2"""
        return self.__W2

    @property
    def b2(self):
        """ return private b2"""
        return self.__b2

    @property
    def A2(self):
        """ return private a2"""
        return self.__A2

    def forward_prop(self, X):
        """ set A with forward prop"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ return the cost """
        cost = -(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))).mean()
        return cost

    def evaluate(self, X, Y):
        """ evaluate neuron prediction """
        self.forward_prop(X)
        return np.round(self.__A2).astype(int), self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ calculate gradient descent"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = dz2.mean(axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = dz1.mean(axis=1, keepdims=True)
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
