#!/usr/bin/env python3
""" normal class """


class Normal:
    """ creating class Normal"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """ init function to initialize class """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum(map(lambda n: pow(n - self.mean,
                                2), data)) / len(data)) ** .5)

    def z_score(self, x):
        """ calculate Z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ calculate x value from z score"""
        return (z * self.stddev) + self.mean
