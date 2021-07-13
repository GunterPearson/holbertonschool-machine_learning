#!/usr/bin/env python3
import numpy as np

def matrix(matrix):
    pad_arr = np.pad(matrix, ((1, 1), (1, 1)), 'constant') 
    print(pad_arr)


if __name__ == "__main__":
    array = np.zeros((28, 28))
    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x][y] = y + x
    matrix(array)
