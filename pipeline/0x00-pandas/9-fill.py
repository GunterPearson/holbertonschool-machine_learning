#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# print(df.isna().sum())
df = df.drop("Weighted_Price", axis=1)
df.Close.fillna(method="ffill", inplace=True)
df.High.fillna(df.Close, inplace=True)
df.Low.fillna(df.Close, inplace=True)
df.Open.fillna(df.Close, inplace=True)
df.fillna(0, inplace=True)

print(df.head())
print(df.tail())
