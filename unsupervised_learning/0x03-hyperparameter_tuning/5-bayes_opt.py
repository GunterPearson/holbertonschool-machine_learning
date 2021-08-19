#!/usr/bin/env python3
"""hyperparameter tuning"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """class constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu - Y_sample - self.xsi
        with np.errstate(divide='warn'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """optimizes the black-box function"""
        all_next = []
        for i in range(iterations):
            x_next, _ = self.acquisition()
            if x_next in all_next:
                break
            y_next = self.f(x_next)
            self.gp.update(x_next, y_next)
            all_next.append(x_next)
        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        self.gp.X = self.gp.X[:-1]
        x_next = self.gp.X[index]
        y_next = self.gp.Y[index]
        return x_next, y_next
