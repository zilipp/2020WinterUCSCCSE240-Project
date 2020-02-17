import os
import random

# import keras
import pandas as pd
import numpy as np
import datetime
import setuptools
import time


import matplotlib.pyplot as plt  # to graphics plot
import seaborn as sns  # a good library to graphic plots

color = sns.color_palette()
import squarify  # to better understand proportion of categorys - it's a treemap layout algorithm

# Importing librarys to use on interactive graphs
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

import json  # to convert json in df
from pandas import json_normalize  # to normalize the json file

# machine learning models
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from keras.losses import mean_squared_error as mse_loss

from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


 # 1. load data to df, after parsing jason
df_train = pd.read_csv("../data/train_concise.csv")
df_test = pd.read_csv("../data/test_concise.csv")
print(df_train.info())
print(df_test.info())

df_train['date'] = pd.to_datetime(
    df_train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['weekday'] = df_train['date'].dt.weekday
df_train['weekofyear'] = df_train['date'].dt.weekofyear
# df_train.drop(['date'], axis=1, errors='ignore', inplace=True)

df_test['date'] = pd.to_datetime(
    df_test['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['weekday'] = df_test['date'].dt.weekday
df_test['weekofyear'] = df_test['date'].dt.weekofyear
# df_test.drop(['date'], axis=1, errors='ignore', inplace=True)
df_train.head()

no_use = ["date", "fullVisitorId", 'totalstransactionRevenue']
cat_cols = ['channelGrouping',
        'deviceoperatingSystem',
        'geoNetworkcity', 'geoNetworkcontinent',
        'geoNetworkcountry', 'geoNetworkmetro',
        'geoNetworknetworkDomain', 'geoNetworkregion',
        'trafficSourcemedium', 'trafficSourcekeyword',
        'trafficSourcesource', 'trafficSourcereferralPath',
        'devicebrowser', 'geoNetworksubContinent', 'devicedeviceCategory',
        'month', 'day', 'weekday', 'weekofyear']
num_cols = ['visitNumber', 'visitStartTime', 'totalshits', 'totalspageviews']
max_values = {}
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))
    df_test[col] = lbl.transform(list(df_test[col].values.astype('str')))
    max_values[col] = max(df_train[col].max(), df_test[col].max())  + 2


# df_test.drop(['date'], axis=1, errors='ignore', inplace=True)
df_train = df_train.sort_values('date')
train_X = df_train.drop(no_use, axis=1)
train_y = df_train['totalstransactionRevenue']
test_X = df_test.drop([col for col in no_use if col in df_test.columns], axis=1)
n_fold = 10
folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)


# converting data to format which can be used by Keras
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols
    X = {col: np.array(df[col]) for col in cols}
    print("Data ready for Vectorization")

    return X

test_X_keras = get_keras_data(test_X, num_cols, cat_cols)


def train_model(keras_model, X_t, y_train, batch_size, epochs, X_v, y_valid, reduce_lr=False, patience=3):
    """
    Helper function to train model. Also I noticed that ReduceLROnPlateau is rarely
    useful, so added an option to turn it off.
    """

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint("model.hdf5",
                                       save_best_only=True, verbose=1, monitor='val_root_mean_squared_error',
                                       mode='min')
    if reduce_lr:
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.000005, verbose=1)
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                               validation_data=(X_v, y_valid), verbose=False,
                               callbacks=[early_stopping, model_checkpoint, reduce_lr])

    else:
        hist = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                               validation_data=(X_v, y_valid), verbose=False,
                               callbacks=[early_stopping, model_checkpoint])

    keras_model = load_model("model.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

    return keras_model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))

scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X)):
    print('Fold:', fold_n)
    X_train, X_valid = train_X.iloc[train_index], train_X.iloc[valid_index]
    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]
    X_t = get_keras_data(X_train, num_cols, cat_cols)
    X_v = get_keras_data(X_valid, num_cols, cat_cols)

    # Neural network
    model = Sequential()
    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

    hist = model.fit(X_train, y_train, batch_size=100, epochs=1, validation_data=(X_valid, y_valid))

