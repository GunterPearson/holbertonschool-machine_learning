#!/usr/bin/env python3
"""preprocess time series data"""
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


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
    print("TOP 5 ROWS AND SHAPE")
    print(bit_df.head())
    print(bit_df.shape)
    print("********************")
    print('')

    # let's drop the Timestamp column
    print("UPDATE TIMESTAMP FROM UNIX TO DATETIME")
    bit_df["Timestamp"] = pd.to_datetime(bit_df["Timestamp"], unit='s')
    print(bit_df.head())
    print("")

    # view amount of NAN per category
    print("TOTAL NAN VALUES PER CATEGORY:")
    print(bit_df.isna().sum())
    print('')

    # drop all NAN
    # df = bit_df.dropna()
    # OR forward fill all missing NAN values
    print("FORWARD FILL NAN VALUES:")
    df = bit_df.ffill()
    print(df.head())
    print("********************")

    # convert minutes to hours
    print("CONVERT MINUTES TO HOURS")
    data = df.copy()
    data = data[8::60]
    print(data.head())
    print("********************")

    # show correlation to know what cols to drop
    print("VIEW CORRELATION MATRIX TO REMOVE COLUMNS HIGHLY CORRELATED")
    correlation = data.corr()
    print(correlation)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True)
    plt.show()
    print("********************")

    # Droping Highly correlated feature cols
    print("DROPING HIGHLY CORRELATED FEATURE COLUMNS AND CHANGING INDEX TO TIME")
    # data = data.drop(columns=['Open', 'High', 'Low', 'Weighted_Price', ])
    # MAKES DATA UNIVARIATE
    data = data[['Timestamp', 'Close']]
    data = data.set_index('Timestamp')
    print(data.head())
    print("********************")

    # Show closing prices by year
    plt.figure(figsize=(12, 8))
    plt.plot(data["Close"])
    plt.ylabel("Close")
    plt.xlabel("Year")
    plt.title("Close Price")
    plt.show()


    # plot for feel of data
    # df.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # visualize Update data
    # plot_cols = df.columns.to_list()
    # plot_features = df[plot_cols]
    # plot_features.index = date_time
    # _ = plot_features.plot(subplots=True)
    # plt.show()
    return data

def split_data(data, split_size=0.8):
    """split data into validation train and test set"""
    n = len(data)
    print(n)
    print(data.shape)
    train_data = data[:int(n * split_size)]
    test_data = data[int(n * split_size):]

    return train_data, test_data

    
def plot_data(train, test):
    """ plot train and test data"""
    plt.figure(figsize=(12, 8))
    plt.scatter(x=train.index, y=train["Close"], s=5, label="Train Data")
    plt.scatter(x=test.index, y=test["Close"], s=5, label="Test Data")
    plt.legend()
    plt.show()


def normalize(train, test):
    """normalize data"""
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std

    test = (test - train_mean) / train_std
    return train, test

INPUT_WIDTH = 24
LABEL_WIDTH = 1
BATCH_SIZE = 64
EPOCHS = 5

def split_window(data):
    """split data into windows"""
    input_slice = slice(0, INPUT_WIDTH)
    output_slice = slice(INPUT_WIDTH, None)
    # [batch_size, window-timestamp, features]
    inputs = data[:, input_slice, :]
    outputs = data[:, output_slice, :]
    return inputs, outputs

def make_dataset(data):
    """make dataset and map to split window"""
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=INPUT_WIDTH + LABEL_WIDTH,
        sequence_stride=1,
        shuffle=True,
        batch_size=BATCH_SIZE
    )
    ds = ds.map(split_window)
    return ds

def compile_and_fit(model, train_ds, test_ds, patience=2):
    """compile and fit model"""
    model.compile(
        loss=tf.losses.MeanAbsoluteError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[early_stopping]
    )
    return history

def plot_performance(history):
    """plot performance of train and test"""
    # for training
    plt.plot(history.history['mean_absolute_error'])
    # for test
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel("mean absolute error")
    plt.xlabel("epoch")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_graph(model, train_ds):
    """plot """
    plt.figure(figsize=(10, 7))
    for b, (x, y) in enumerate(train_ds.take(4)):
        plt.subplot(2, 2, b+1)
        prediction = model.predict(x)[b]
        plt.plot(list(range(INPUT_WIDTH)), x[b, :, -1]) # current data
        plt.plot(INPUT_WIDTH, y[b].numpy(), 'r*', label="Label") # Label
        plt.plot(INPUT_WIDTH, prediction, 'g^', label='Prediction') # prediction
        plt.show()

if __name__ == "__main__":
    data = preprocess()
    train, test = split_data(data)
    plot_data(train, test)
    train, test = normalize(train, test)
    train_ds = make_dataset(train)
    test_ds = make_dataset(test)
    # for i, j in train_ds.take(1):
    #     print(i.shape)
    #     print(i)
    #     print(j.shape)
    #     print(j)
    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    deep_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    history = compile_and_fit(simple_model, train_ds, test_ds)
    plot_performance(history)
    plot_graph(simple_model, train_ds)
