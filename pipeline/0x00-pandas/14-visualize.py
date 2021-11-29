#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# The column Weighted_Price should be removed
df.drop("Weighted_Price", axis=1, inplace=True)
# Rename the column Timestamp to Date
# Convert the timestamp values to date values
df.rename(columns={"Timestamp": "Date"}, inplace=True)
df.Date = pd.to_datetime(df.Date, unit='s')
# Index the data frame on Date
df.set_index("Date", inplace=True)
# Missing values in Close should be set to the previous row value
df.Close.fillna(method='ffill', inplace=True)
# Missing values in High, Low, Open should be set to the same row’s Close value
df.High.fillna(df.Close, inplace=True)
df.Low.fillna(df.Close, inplace=True)
df.Open.fillna(df.Close, inplace=True)
df.fillna(0, inplace=True)
# Plot the data from 2017 and beyond at daily intervals and group by day
# - High: max
# - Low: min
# - Open: mean
# - Close: mean
# - Volume(BTC): sum
# - Volume(Currency): sum

df = df.loc['2017-01-01':]
df = df.resample('D').agg(
    {'Open': 'mean',
     'High': 'max',
     'Low': 'min',
     'Close': 'mean',
     'Volume_(BTC)': 'sum',
     'Volume_(Currency)': 'sum'})

df.plot()
plt.show()
