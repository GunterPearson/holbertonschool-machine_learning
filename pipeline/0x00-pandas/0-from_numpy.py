#!/usr/bin/env python3
"""from numpy to pandas"""
import pandas as pd
import string


def from_numpy(array):
    """that creates a pd.DataFrame from a np.ndarray"""
    let = string.ascii_uppercase
    print(pd.DataFrame(array, columns=[let[x] for x in range(len(array[0]))]))
