#!/usr/bin/env python3
"""hyperparameter tuning"""
import numpy as np


class GaussianProcess():
    """GaussianProcess"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """represents a noiseless 1D Gaussian process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix"""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist2 = sqdist - 2 * np.dot(X1, X2.T)
        result = self.sigma_f ** 2 * np.exp(-0.5 / self.l**2 * sqdist2)
        return result
