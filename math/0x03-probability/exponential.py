#!/usr/bin/env python3
""" exponential"""


class Exponential:
    """ create exponential class"""

    def __init__(self, data=None, lambtha=1.):
        """ init function for Poisson class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """ find the probability distributions """
        e = 2.7182818285
        if x < 0:
            return 0
        return self.lambtha * pow(e, -(self.lambtha * x))

    def cdf(self, x):
        """ find the continous distribution """
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - pow(e, -(self.lambtha * x))
