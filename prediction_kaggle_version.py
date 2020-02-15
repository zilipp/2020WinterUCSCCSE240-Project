# working Kaggle version
#https://www.kaggle.com/sherrymay/aiprediction-ga-customer-revenue
#Woring at large scale with over 2G data, kaggle provide the GPU

import os
import random
import pandas as pd
import numpy as np
import datetime
import setuptools
import time

from scipy.stats import kurtosis, skew  # it's to explore some statistics of numerical value

import matplotlib.pyplot as plt  # to graphics plot
import seaborn as sns  # a good library to graphic plots

color = sns.color_palette()
import squarify  # to better understand proportion of categorys - it's a treemap layout algorithm

# Importing librarys to use on interactive graphs
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

from plotly import tools
#from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

import json  # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file

# machine learning models
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

# sel defined class
#from src.visualization import Visualization

# credit to:
# parse JSON:
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
# baseline model : lgbm
# https://www.kaggle.com/zili100097/simple-exploration-baseline-ga-customer-revenue/edit


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import json  # to convert json in df
#from pandas import json_normalize  
from pandas.io.json import json_normalize # to normalize the json file
import random
import datetime
from sklearn import preprocessing
import os
import random
import pandas as pd
import numpy as np
import datetime
import setuptools
import time

from scipy.stats import kurtosis, skew  # it's to explore some statistics of numerical value

import matplotlib.pyplot as plt  # to graphics plot
import seaborn as sns  # a good library to graphic plots

color = sns.color_palette()
import squarify  # to better understand proportion of categorys - it's a treemap layout algorithm

# Importing librarys to use on interactive graphs
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

from plotly import tools
#from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

import json  # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file

# machine learning models
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

#pd.read_csv('../input/train.csv').head()
# Any results you write to the current directory are saved as output.


dir_path = "../input/"
p = 0.01  # fractional number to skip rows and read just a random sample of the our dataset.
plt.style.use('fivethirtyeight')  # to set a style to all graphs

#p = 1  # fractional number to skip rows and read just a random sample of the our dataset.
# Transform the json format columns in table
def json_read(df):
    # joining the [ path + df received]
    data_frame = dir_path + df
    columns = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format

    df = pd.read_csv(data_frame,
                     converters={column: json.loads for column in columns},  # loading the json columns properly
                     dtype={'fullVisitorId': 'str'},  # transforming this column to string
                     skiprows=None #lambda i: i > 0 and random.random() > p
                     )  # Number of rows that will be imported randomly

    for column in columns:
        # It will normalize and set the json to a table
        column_as_df = json_normalize(df[column])
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}{subcolumn}" for subcolumn in column_as_df.columns]
        column_as_df.columns = [subcolumn.replace('.', '') for subcolumn in column_as_df.columns]
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        print(list(column_as_df.columns.values))
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df

def drop_features(train, test):
    to_drop = ['sessionId', 'socialEngagementType', 'devicebrowserVersion', 'devicebrowserSize', 'deviceflashVersion',
               'devicelanguage',
               'devicemobileDeviceBranding', 'devicemobileDeviceInfo', 'devicemobileDeviceMarketingName',
               'devicemobileDeviceModel',
               'devicemobileInputSelector', 'deviceoperatingSystemVersion', 'devicescreenColors',
               'devicescreenResolution',
               'geoNetworkcityId', 'geoNetworklatitude', 'geoNetworklongitude', 'geoNetworknetworkLocation',
               'trafficSourceadwordsClickInfocriteriaParameters', 'trafficSourceadwordsClickInfogclId',
               'trafficSourcecampaign',
               'trafficSourceadwordsClickInfopage', 'trafficSourcereferralPath',
               'trafficSourceadwordsClickInfoslot',
               'trafficSourceadContent', 'trafficSourcekeyword', 'trafficSourceadwordsClickInfoadNetworkType',
               'totalsbounces', 'totalsnewVisits', 'totalsvisits',
               'trafficSourceisTrueDirect',
               'trafficSourceadwordsClickInfoisVideoAd', 'totalsvisits']
    result_df_test = test.drop(to_drop, axis=1)
    
    #train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
    if 'trafficSourcecampaignCode' in train.columns:
        result_df_train = train.drop(to_drop + ['trafficSourcecampaignCode'], axis=1)
    else:
        result_df_train = train.drop(to_drop , axis=1)
    return result_df_train, result_df_test

def change_feature_type_and_fill_na(train, test):
    def common(df):
        df.loc[df['geoNetworkcity'] == "(not set)", 'geoNetworkcity'] = np.nan
        df.loc[df['geoNetworkcity'] == "not available in demo dataset", 'geoNetworkcity'] = np.nan
        df['geoNetworkcity'].fillna("NaN", inplace=True)

        df.loc[df['geoNetworkmetro'] == "(not set)", 'geoNetworkmetro'] = np.nan
        df.loc[df['geoNetworkmetro'] == "not available in demo dataset", 'geoNetworkmetro'] = np.nan
        df['geoNetworkmetro'].fillna("NaN", inplace=True)

        df.loc[df['geoNetworknetworkDomain'] == "not available in demo dataset", 'geoNetworknetworkDomain'] = np.nan
        df['geoNetworknetworkDomain'].fillna("NaN", inplace=True)

        df.loc[df['geoNetworkregion'] == "not available in demo dataset", 'geoNetworkregion'] = np.nan
        df['geoNetworkregion'].fillna("NaN", inplace=True)

        df["totalshits"] = df["totalshits"].astype(int)  # setting numerical to int

        df['totalspageviews'].fillna(1, inplace=True)  # filling NA's with 1
        df['totalspageviews'] = df['totalspageviews'].astype(int)  # setting numerical column as integer

        return df

    result_df_train = common(train)
    result_df_test = common(test)
    result_df_train["totalstransactionRevenue"] = result_df_train["totalstransactionRevenue"].fillna(0.0).astype \
        (float)
    return result_df_train, result_df_test

def category_to_number(train, test):
    cat_cols = ["channelGrouping", "devicebrowser",
                "devicedeviceCategory", "deviceoperatingSystem",
                "geoNetworkcity", "geoNetworkcontinent",
                "geoNetworkcountry", "geoNetworkmetro",
                "geoNetworknetworkDomain", "geoNetworkregion",
                "geoNetworksubContinent",
                "trafficSourcemedium",
                "trafficSourcesource"]

    for col in cat_cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    return train, test

#dir_path = "../output/"


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
    print("features inside separate data: ")
    print(features)
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


def validate(val_df, pred_val):
    val_pred_df = pd.DataFrame({"fullVisitorId": val_df["fullVisitorId"].values})
    val_pred_df["transactionRevenue"] = val_df["totalstransactionRevenue"].values
    val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
    # print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
    print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values),
                                             np.log1p(val_pred_df["PredictedRevenue"].values))))


#if __name__ == '__main__':
    # 1. load data to df, after parsing jason
    #df_train = pd.read_csv("train_concise.csv")
    #df_test = pd.read_csv("test_concise.csv")
    
if __name__ == '__main__':
    # 1. load data to df, and parse jason, then output to csv
    df_train = json_read("train.csv")
    df_test = json_read("test.csv")

    # process data feature
    # df_train['date'] = df_train['date'].apply(
    #     lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

    # drop features
    df_train, df_test = drop_features(df_train, df_test)
    print("df_train after drop features")
    print(df_train)
    # fill na
    df_train, df_test = change_feature_type_and_fill_na(df_train, df_test)

    # category to number
    df_train, df_test = category_to_number(df_train, df_test)
    print(df_train.info())
    print(df_test.info())
    # # add index name
    # df_train.rename(index={0: "index"})
    # df_test.rename(index={0: "index"})

    #df_train.to_csv('train_concise.csv', index=False)
    #df_test.to_csv('test_concise.csv', index=False)
    print(df_train.head())
    print(df_test.head())
    # print(df_train['date'].head())
    # df_train['date'] = pd.to_datetime(df_train['date'], format='%Y%M%d')
    # process data feature
    # df_train['date'] = df_train['date'].apply(
    #     lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    #print(df_train.info())
    #print(df_test.info())

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

    print(train_X.info())

    # build and train model
    pred_test, model, pred_val = run_lgb(train_X, train_y, val_X, val_y, test_X)

    # validate the model
    validate(val_df, pred_val)