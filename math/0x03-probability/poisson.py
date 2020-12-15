#!/usr/bin/env python3
""" poisson """


class Poisson:
    """ created poisson class"""

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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ calculate the pmf"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        return (pow(e, self.lambtha * -1) * pow(self.lambtha, k)
                / factorial(k))

    def cdf(self, k):
        """ calculate cdf"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        return sum([self.pmf(x) for x in range(k + 1)])


def factorial(k):
    """ return factorial"""
    if k < 0:
        return None
    if k == 0:
        return 1
    if k == 1:
        return 1
    return k * factorial(k - 1)
