#!/usr/bin/env python3

    def pdf(self, x):
        """ return pdf"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        n = x.shape[0]
        m = self.mean
        c = self.cov
        t = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(c))
        j = np.linalg.inv(c)
        ran = (-0.5 * np.matmul(np.matmul((x - m).T, j), x - self.mean))
        pdf = (1 / t) * np.exp(ran[0][0])
        return pdf
