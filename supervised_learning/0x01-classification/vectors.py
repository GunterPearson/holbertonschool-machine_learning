#!/usr/bin/env python3
""" test.py """
import numpy as np
import time


a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
z = -(a * np.log(b) + (1 - a) * (np.log(1.0000001 - b))).mean()
toc = time.time()
print(z)
print("VECOTRIZED: " + str(1000 * (toc - tic)) + "ms")


tic = time.time()
x = -(a * np.log(b) + (1 - a) * (np.log(1.0000001 - b))) / len(a)
toc = time.time()
print(x)
print("NONVECOTRIZED: " + str(1000 * (toc - tic)) + "ms")
