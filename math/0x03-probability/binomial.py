#!/usr/bin/env python3
""" binomial class """


class Binomial:
    """ creating binomial class """

    def __init__(self, data=None, n=1, p=0.5):
        """ initialazition of binomial class """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            var = float((sum(map(lambda n: pow(n - mean,
                        2), data)) / len(data)))
            self.p = - (var / mean) + 1
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """ calculates probabilit mass """
        if k < 0:
            return 0
        k = int(k)
        n = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return (pow(self.p, k) * pow(1 - self.p, self.n - k)) * n

    def cdf(self, k):
        """ calculate the cummalitive distribution"""
        if k < 0:
            return 0
        k = int(k)
        return sum(map(self.pmf, range(k + 1)))


def factorial(k):
    """ return factorial"""
    if k < 0:
        return None
    if k == 0:
        return 1
    if k == 1:
        return 1
    return k * factorial(k - 1)
