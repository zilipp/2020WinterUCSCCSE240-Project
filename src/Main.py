import os
import random

# import keras
import pandas as pd
import numpy as np
import datetime
import setuptools
import time

# from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import kurtosis, skew  # it's to explore some statistics of numerical value

import matplotlib.pyplot as plt  # to graphics plot
import seaborn as sns  # a good library to graphic plots

color = sns.color_palette()
import squarify  # to better understand proportion of categorys - it's a treemap layout algorithm

# Importing librarys to use on interactive graphs
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

from plotly import tools
from plotly import subplots
import plotly.offline as py
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

# sel defined class
from src.visualization import Visualization

# credit to:
# parse JSON:
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
# baseline model : lgbm
# https://www.kaggle.com/zili100097/simple-exploration-baseline-ga-customer-revenue/edit


dir_path = "../data/"
p = 0.001  # fractional number to skip rows and read just a random sample of the our dataset.
mod = 'XGBOOST'#'LGBM' # RNN / XGBOOST
plt.style.use('fivethirtyeight')  # to set a style to all graphs


def missing_value_info(data):
    columns = data.columns[data.isnull().any()].tolist()
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending=False)
    # getting the percent and order of null
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)

    # Concatenating the total and percent
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("Total columns at least one Values: ")
    print(df[~(df['Total'] == 0)])  # Returning values of nulls different of 0

    print("\n Total of Sales % of Total: ", round((df_train[df_train['totalstransactionRevenue'] != np.nan][
                                                       'totalstransactionRevenue'].count() / len(
        df_train['totalstransactionRevenue']) * 100), 4))

    return columns


def revenue_customers(train_df, test_df):
    train_df["totalstransactionRevenue"] = train_df["totalstransactionRevenue"].astype('float')
    gdf = train_df.groupby("fullVisitorId")["totalstransactionRevenue"].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totalstransactionRevenue"].values)))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('TransactionRevenue', fontsize=12)
    # plt.show()

    nzi = pd.notnull(train_df["totalstransactionRevenue"]).sum()
    nzr = (gdf["totalstransactionRevenue"] > 0).sum()
    # print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])
    # print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])
    #
    # print("Number of unique visitors in train set : ", train_df.fullVisitorId.nunique(), " out of rows : ",
    #       train_df.shape[0])
    # print("Number of unique visitors in test set : ", test_df.fullVisitorId.nunique(), " out of rows : ",
    #       test_df.shape[0])
    # print("Number of common visitors in train and test set : ",
    #       len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique()))))

    return gdf


def separate_data(train, test):
    features = list(train.columns.values.tolist())
    features.remove("totalstransactionRevenue")
    features.remove("fullVisitorId")
    features.remove("date")

    # Split the train dataset into development and valid based on time
    train['date'] = train['date'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    test['date'] = test['date'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

    dev_df = train[train['date'] <= datetime.date(2017, 5, 31)]
    val_df = train[train['date'] > datetime.date(2017, 5, 31)]
    dev_y = np.log1p(dev_df["totalstransactionRevenue"].values)
    val_y = np.log1p(val_df["totalstransactionRevenue"].values)

    dev_X = dev_df[features]
    val_X = val_df[features]
    test_X = test[features]

    return dev_X, dev_y, val_X, val_y, test_X, dev_df, val_df


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        # max number of leaves in one tree
        "num_leaves": 30,
        # minimal number of data in one lea
        "min_child_samples": 100,
        "learning_rate": 0.1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.5,
        "bagging_freq": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y[pred_test_y < 0] = 0

    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    pred_val_y[pred_val_y < 0] = 0
    return pred_test_y, model, pred_val_y


# def run_NN(train_X, train_y, val_X, val_y, test_X):
#     # train_X = K.cast_to_floatx(train_X)
#     # train_y = K.cast_to_floatx(train_y)
#     # val_X = K.cast_to_floatx(val_X)
#     # val_y = K.cast_to_floatx(val_y)
#
#     # Neural network
#     # model = Sequential()
#     # model.add(Dense(30, input_dim=len(train_X[0]), activation='relu'))
#     # model.add(Dense(40, activation='relu'))
#     # model.add(Dense(12, activation='relu'))
#     # model.add(Dense(1, activation='linear'))
#
#     model = keras.Sequential([
#         layers.Dense(30, activation='relu', input_shape=[len(train_X[0])]),
#         layers.Dense(25, activation='relu'),
#         layers.Dense(1)
#     ])
#
#     model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
#     hist = model.fit(train_X, train_y, batch_size=30, epochs=15, validation_data=(val_X, val_y))
#     pred_test = model.predict([test_X], batch_size=30, verbose=1)
#     return pred_test


def validate(val_df, pred_val):
    val_pred_df = pd.DataFrame({"fullVisitorId": val_df["fullVisitorId"].values})
    val_pred_df["transactionRevenue"] = val_df["totalstransactionRevenue"].values
    val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
    val_pred_df = val_pred_df.groupby('fullVisitorId')[['transactionRevenue', 'PredictedRevenue']].sum().reset_index()
    print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df['transactionRevenue'].values),
                                             np.log1p(val_pred_df['PredictedRevenue'].values))))

def show_feature_importance(model):
    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()


def run_xgb(train_X, train_y, val_X, val_y, test_X):

    # fit model no training data
    model = XGBClassifier()
    model.fit(train_X, train_y)

    y_pred_val = model.predict(val_X)
    y_pred_val = [round(value) for value in y_pred_val]
    y_pred_val = [0 if i < 0 else i for i in y_pred_val]

    y_pred_test = model.predict(test_X)
    y_pred_test = [round(value) for value in y_pred_test]
    y_pred_test = [0 if i < 0 else i for i in y_pred_test]
    return y_pred_test, model, y_pred_val


if __name__ == '__main__':
    # 1. load data to df, after parsing jason
    df_train = pd.read_csv("../data/train_concise.csv")
    df_test = pd.read_csv("../data/test_concise.csv")

    print(df_train.info())
    print(df_test.info())

    # group data frame by fullVisitorId
    # gdf = revenue_customers(df_train, df_test)

    # # 2.data visualization(this part only plot, do not change data)
    # vis = Visualization(df_train)
    # # a) with time
    # vis.plot_revenue_count_with_time()
    # # b) difference of device
    # vis.plot_diff_device_importance()
    # # c) traffic source
    # vis.plot_diff_traffic_importance()
    # # d) geo distribution
    # vis.plot_diff_geo_importance()
    # # e) visit profile
    # vis.plot_visit_importance()

    # separate labels and split data
    train_X, train_y, val_X, val_y, test_X, dev_df, val_df = separate_data(df_train, df_test)
    print('==========final data==========')
    print(train_X.shape)
    print(train_y.shape)
    print(val_X.shape)
    print(val_y.shape)
    print(test_X.shape)

    # build and train model
    if mod == 'LGBM':
        pred_test, model, pred_val = run_lgb(train_X, train_y, val_X, val_y, test_X)
        # validate the model
        validate(val_df, pred_val)
        # feature importance
        show_feature_importance(model)
    # elif mod == 'NN':
    #     pred_test = run_NN(train_X, train_y, val_X, val_y, test_X)
    #     print('NN done')
    elif mod == 'XGBOOST':
        pred_test, model, pred_val = run_xgb(train_X, train_y, val_X, val_y, test_X)
        validate(val_df, pred_val)
        print('XGBOOST done')




