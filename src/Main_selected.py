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
import json  # to convert json in df
from pandas import json_normalize  # to normalize the json file

# machine learning models
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

#
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# import keras.backend as K

from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# sel defined class
from src.visualization import Visualization

# credit to:
# parse JSON:
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
# baseline model : lgbm
# https://www.kaggle.com/zili100097/simple-exploration-baseline-ga-customer-revenue/edit


dir_path = "../data/"
p = 0.001  # fractional number to skip rows and read just a random sample of the our dataset.
mod = 'CBOOST' #'LGBOOST' # RNN / XGBOOST /'CBOOST'
plt.style.use('fivethirtyeight')  # to set a style to all graphs
cat_cols = ['channelGrouping',
                'deviceoperatingSystem',
                'geoNetworkcity', 'geoNetworkcontinent',
                'geoNetworkcountry', 'geoNetworkmetro',
                'geoNetworknetworkDomain', 'geoNetworkregion',
                'trafficSourcemedium', 'trafficSourcekeyword',
                'trafficSourcesource', 'trafficSourcereferralPath',
                'devicebrowser', 'geoNetworksubContinent', 'devicedeviceCategory']


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


def category_to_number(train, test):
    # cat_cols = ['channelGrouping',
    #             'deviceoperatingSystem',
    #             'geoNetworkcity', 'geoNetworkcontinent',
    #             'geoNetworkcountry', 'geoNetworkmetro',
    #             'geoNetworknetworkDomain', 'geoNetworkregion',
    #             'geoNetworknetworkDomain',
    #             'trafficSourcemedium', 'trafficSourcekeyword',
    #             'trafficSourcesource', 'trafficSourcereferralPath',
    #             'devicebrowser', 'geoNetworksubContinent', 'devicedeviceCategory']

    for col in cat_cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    return train, test


def separate_data(train, test):
    features = list(train.columns.values.tolist())
    features.remove("totalstransactionRevenue")
    features.remove("fullVisitorId")
    features.remove("date")
    # =================selected================
    features.remove("trafficSourcekeyword")
    features.remove("geoNetworknetworkDomain")
    features.remove("trafficSourcereferralPath")
    features.remove("devicebrowser")
    features.remove("trafficSourcemedium")
    features.remove("geoNetworkregion")
    features.remove("channelGrouping")
    features.remove("geoNetworksubContinent")
    features.remove("geoNetworkcontinent")
    features.remove("devicedeviceCategory")
    features.remove("geoNetworkcity")
    features.remove("deviceoperatingSystem")
    features.remove("deviceisMobile")
    # =================selected================

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

    print('==========final data==========')
    print('train_X shape ' + str(dev_X.shape))
    print('train_y shape ' + str(dev_y.shape))
    print('val_X shape ' + str(val_X.shape))
    print('val_y shape ' + str(val_y.shape))
    print('test_X shape ' + str(test_X.shape))

    return dev_X, dev_y, val_X, val_y, test_X, dev_df, val_df


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

def cboost_feature_importance(model):
    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()



# ===================== models ============================

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

    model = lgb.train(params, lgtrain, 400, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y[pred_test_y < 0] = 0

    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    pred_val_y[pred_val_y < 0] = 0
    return pred_test_y, model, pred_val_y


def run_cb(train_X, train_y, val_X, val_y, test_X):
    # Index(['channelGrouping', 'visitNumber', 'visitStartTime', 'devicebrowser',
    #        'deviceoperatingSystem', 'deviceisMobile', 'devicedeviceCategory',
    #        'geoNetworkcontinent', 'geoNetworksubContinent', 'geoNetworkcountry',
    #        'geoNetworkregion', 'geoNetworkmetro', 'geoNetworkcity',
    #        'geoNetworknetworkDomain', 'totalshits', 'totalspageviews',
    #        'trafficSourcesource', 'trafficSourcemedium',
    #        'trafficSourcereferralPath', 'trafficSourcekeyword'],
    #       dtype='object')

    # selected
    # Index(['visitNumber', 'visitStartTime', 'geoNetworkcountry', 'geoNetworkmetro',
    #        'totalshits', 'totalspageviews', 'trafficSourcesource'],
    #       dtype='object')


    # categorical_features_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]
    # selected
    categorical_features_indices = [2, 3, 6]


    model = CatBoostRegressor(iterations=1000,
                              learning_rate=0.05,
                              depth=10,
                              eval_metric='RMSE',
                              random_seed=42,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=50,
                              od_wait=20)
    model.fit(train_X, train_y,
              cat_features=categorical_features_indices,
              eval_set=(val_X, val_y),
              use_best_model=True,
              verbose=True)

    pred_test_y = model.predict(test_X)
    pred_val_y = model.predict(val_X)
    pred_val_y[pred_val_y < 0] = 0

    feature_score = pd.DataFrame(list(
        zip(train_X.dtypes.index, model.get_feature_importance(Pool(train_X, label=train_y, cat_features=categorical_features_indices)))),
                                 columns=['Feature', 'Score'])

    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort',
                                              na_position='last')

    plt.rcParams["figure.figsize"] = (12, 7)
    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
    ax.set_title("Catboost Feature Importance Ranking", fontsize=14)
    ax.set_xlabel('')

    rects = ax.patches

    labels = feature_score['Score'].round(2)

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.35, label, ha='center', va='bottom')

    plt.show()

    return pred_test_y, model, pred_val_y


# =================== deep learning =======================
def run_NN(train_X, train_y, val_X, val_y, test_X):
    # train_X = K.cast_to_floatx(train_X)
    # train_y = K.cast_to_floatx(train_y)
    # val_X = K.cast_to_floatx(val_X)
    # val_y = K.cast_to_floatx(val_y)

    # Neural network
    # model = Sequential()
    # model.add(Dense(30, input_dim=len(train_X[0]), activation='relu'))
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dense(1, activation='linear'))

    # model = keras.Sequential([
    #     layers.Dense(30, activation='relu', input_shape=[len(train_X[0])]),
    #     layers.Dense(25, activation='relu'),
    #     layers.Dense(1)
    # ])
    #
    # model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    # hist = model.fit(train_X, train_y, batch_size=30, epochs=15, validation_data=(val_X, val_y))
    # pred_test = model.predict([test_X], batch_size=30, verbose=1)

    return pred_test


if __name__ == '__main__':
    # 1. load data to df, after parsing jason
    df_train = pd.read_csv("../data/train_full.csv", low_memory=False)
    df_test = pd.read_csv("../data/test_full.csv", low_memory=False)
    print(df_train.info())
    print(df_test.info())

    # separate labels and split data
    df_train, df_test = category_to_number(df_train, df_test)
    train_X, train_y, val_X, val_y, test_X, dev_df, val_df = separate_data(df_train, df_test)
    print(train_X.columns)

    # build and train model
    if mod == 'LGBOOST':
        pred_test, model, pred_val = run_lgb(train_X, train_y, val_X, val_y, test_X)
        validate(val_df, pred_val)
        show_feature_importance(model)
        print('LGBM done')
    elif mod == 'XGBOOST':
        pred_test, model, pred_val = run_xgb(train_X, train_y, val_X, val_y, test_X)
        validate(val_df, pred_val)
        print('XGBOOST done')
    elif mod == 'CBOOST':
        pred_test, model, pred_val = run_cb(train_X, train_y, val_X, val_y, test_X)
        validate(val_df, pred_val)
        print('CBOOST done')






