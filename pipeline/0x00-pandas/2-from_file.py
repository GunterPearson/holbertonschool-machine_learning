#!/usr/bin/env python3
"""from numpy to pandas"""
import pandas as pd


def from_file(filename, delimiter):
    """returns panda dataframe from csv file"""
    csv = pd.read_csv(filename, sep=delimiter)
    return csv
