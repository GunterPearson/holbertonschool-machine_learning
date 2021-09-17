#!/usr/bin/env python3
"""preprocess time series data"""
import pandas as pd
import matplotlib.pyplot as plt

def preprocess():
    """preprocess data"""
    bitstamp = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    # coinbase = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    bit_df = pd.read_csv(bitstamp)
    # coin_df = pd.read_csv(coinbase)

    # get a feel of the type of data and check perecentage of missing values
    # the dataframe is windowed per 60 second so we need to recast for need by windowing hourly
    print("Bitstamp Describe:")
    print(bit_df.describe().transpose())
    print("********************")

    # view amount of NAN per category
    print("TOTAL NAN VALUES PER CATEGORY:")
    print(bit_df.isna().sum())
    print("********************")

    # drop all NAN
    # df = bit_df.dropna()
    # OR forward fill all missing NAN values
    print("FORWARD FILL NAN VALUES:")
    df = bit_df.ffill()
    print(df.isna().sum())
    print("********************")

    # view top 5 rows of data
    print("TOP 5 ROWS OF BIT DATAFRAME")
    print(df.head())
    print("********************")

    # let's drop the Timestamp column
    print("CREATE NEW DATE TIME DATAFRAME FROM TIMESTAMP")
    date_time = pd.to_datetime(df.pop('Timestamp'), unit='s')
    print(date_time.head())
    print("POP TIMESTAMP COLUMN FROM BIT DATAFRAME")
    print(df.head())
    print("********************")

    # plot for feel of data
    # df.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # show correlation to know what cols to drop
    print("VIEW CORRELATION MATRIX TO REMOVE COLUMNS HIGHLY CORRELATED")
    correlation = df.corr()
    print(correlation)
    # plt.matshow(correlation)
    # plt.show()
    print("********************")

    # Droping Highly correlated feature cols
    print("DROPING HIGHLY CORRELATED FEATURE COLUMNS")
    df = df.drop(columns=['Open', 'High', 'Low', 'Close'])
    print(df.head())
    print("********************")

    # visualize Update data
    plot_cols = df.columns.to_list()
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plt.show()

    

    # print(len(df))
    # print(df.shape)


preprocess()
