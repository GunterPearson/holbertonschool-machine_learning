#!/usr/bin/env python3
"""from numpy to pandas"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# new = df.reset_index(drop=True)
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df[['Datetime', 'Close']]

print(df.tail())
