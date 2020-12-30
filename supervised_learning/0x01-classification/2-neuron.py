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
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-z))
        return self.__A
