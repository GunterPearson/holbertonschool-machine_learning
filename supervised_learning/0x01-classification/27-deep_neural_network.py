#!/usr/bin/env python3
""" Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        for x in range(1, self.__L + 1):
            z = np.matmul(self.weights["W" + str(x)], self.__cache[
                "A" + str(x - 1)]) + self.weights["b" + str(x)]
            if x == self.L:
                temp = np.exp(z)
                A = temp / np.sum(temp, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp((-1) * z))
            self.__cache["A" + str(x)] = A
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ return the cost """
        m = Y.shape[1]
        L = Y * np.log(A)
        return (-1/m) * np.sum(L)

    def evaluate(self, X, Y):
        """ evaluate to binary 1 or 0"""
        self.forward_prop(X)
        A = self.cache["A" + str(self.L)]
        R = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return R, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ gradient descent for deepNN"""
        m = Y.shape[1]
        for x in reversed(range(1, self.__L + 1)):
            AN1 = self.__cache["A" + str(x - 1)]
            A0 = self.__cache["A" + str(x)]
            W0 = self.__weights["W" + str(x)]
            if x == self.__L:
                dz = A0 - Y
            else:
                dz = da * (A0 * (1 - A0))
            db = dz.mean(axis=1, keepdims=True)
            dw = np.matmul(dz, AN1.T) / m
            da = np.matmul(W0.T, dz)
            self.__weights['W' + str(x)] -= (alpha * dw)
            self.__weights['b' + str(x)] -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ train deep neural network"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        step_array = list(range(0, iterations + 1, step))
        cost_array = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose and i in step_array:
                cost = self.cost(Y, self.cache["A" + str(self.L)])
                print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(step_array, cost_array, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ save neural network"""
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                x = pickle.load(f)
                return x
        except FileNotFoundError:
            return None
