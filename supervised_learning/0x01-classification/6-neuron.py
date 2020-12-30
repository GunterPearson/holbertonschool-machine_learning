#!/usr/bin/env python3
""" neuron class"""
import numpy as np


class Neuron:
    """ creating neuron class"""

    def __init__(self, nx):
        """ initialize Neuron class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ return private w"""
        return self.__W

    @property
    def b(self):
        """ return private b"""
        return self.__b

    @property
    def A(self):
        """ return private a"""
        return self.__A

    def forward_prop(self, X):
        """ forward propigation """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ use linear regression for cost """
        cost = -(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))).mean()
        return cost

    def evaluate(self, X, Y):
        """ evaluate neuron prediction """
        m = Y.shape[1]
        A = np.ndarray((1, m))
        A = self.forward_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ calculate gradient descent"""
        m = Y.shape[1]
        dz = A - Y
        db = dz.mean()
        dw = np.matmul(X, dz.T) / m
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ train the neuron """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        while iterations > 0:
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
            iterations -= 1
        return self.evaluate(X, Y)
