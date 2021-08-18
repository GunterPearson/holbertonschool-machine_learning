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

    def predict(self, X_s):
        """predicts the mean and standard deviation """
        sigma_y = 1e-8
        K = self.K + sigma_y**2 * np.eye(len(self.X))
        K1 = self.kernel(self.X, X_s)
        K2 = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(K)

        mu = K1.T.dot(K_inv).dot(self.Y)
        cov = K2 - K1.T.dot(K_inv).dot(K1)
        return mu.reshape(-1), cov.diagonal()
