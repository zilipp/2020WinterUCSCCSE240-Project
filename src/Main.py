import os
import random
import pandas as pd
import numpy as np
import datetime
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
from plotly import subplots
import plotly.offline as py
import plotly.graph_objs as go

import json  # to convert json in df
from pandas import json_normalize  # to normalize the json file

# machine learning models
from sklearn import model_selection, preprocessing, metrics

# import lightgbm as lgb

# sel defined class
from src.visualization import Visualization




# credit to:
# parse JSON:
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
# baseline model : lgbm
# https://www.kaggle.com/zili100097/simple-exploration-baseline-ga-customer-revenue/edit


dir_path = "../data/"
p = 0.001  # fractional number to skip rows and read just a random sample of the our dataset.
plt.style.use('fivethirtyeight')  # to set a style to all graphs

# Transform the json format columns in table
def json_read(df):
    # joining the [ path + df received]
    data_frame = dir_path + df
    columns = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format

    df = pd.read_csv(data_frame,
                     converters={column: json.loads for column in columns},  # loading the json columns properly
                     dtype={'fullVisitorId': 'str'},  # transforming this column to string
                     skiprows=lambda i: i > 0 and random.random() > p
                     )  # Number of rows that will be imported randomly

    for column in columns:
        # It will normalize and set the json to a table
        column_as_df = json_normalize(df[column])
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df

def normal_csv_read(df):
    data_frame = dir_path + df
    df = pd.read_csv(data_frame)
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df


def missing_value_info(data):
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending=False)
    # getting the percent and order of null
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)

    # Concatenating the total and percent
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("Total columns at least one Values: ")
    print(df[~(df['Total'] == 0)])  # Returning values of nulls different of 0

    print("\n Total of Sales % of Total: ", round((df_train[df_train['totals.transactionRevenue'] != np.nan][
                                                       'totals.transactionRevenue'].count() / len(
        df_train['totals.transactionRevenue']) * 100), 4))

    return


def revenue_customers(train_df, test_df):
    train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
    gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('TransactionRevenue', fontsize=12)
    plt.show()

    nzi = pd.notnull(train_df["totals.transactionRevenue"]).sum()
    nzr = (gdf["totals.transactionRevenue"] > 0).sum()
    print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])
    print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])

    print("Number of unique visitors in train set : ", train_df.fullVisitorId.nunique(), " out of rows : ",
          train_df.shape[0])
    print("Number of unique visitors in test set : ", test_df.fullVisitorId.nunique(), " out of rows : ",
          test_df.shape[0])
    print("Number of common visitors in train and test set : ",
          len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique()))))

    return gdf


if __name__ == '__main__':
    # # 1. load data to df, and parse jason, then output to csv
    # df_train = json_read("train.csv")
    # df_test = json_read("test.csv")
    # # process data feature
    # df_train['date'] = df_train['date'].apply(
    #     lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    # df_train.to_csv('../data/train_concise.csv')
    # df_test.to_csv('../data/test_concise.csv')

    # or 1. load data to df, after parsing jason
    df_train = normal_csv_read("train_concise.csv")
    df_test = normal_csv_read("test_concise.csv")

    # group data frame by fullVisitorId
    gdf = revenue_customers(df_train, df_test)

    # 2.data visualization(this part only plot, do not change data)
    vis = Visualization(df_train)
    # a) with time
    vis.plot_revenue_count_with_time()
    # b) difference of device
    vis.plot_diff_device_importance()
    # c) traffic source
    vis.plot_diff_traffic_importance()
    # d) geo distribution
    vis.plot_diff_geo_importance()


    # 3 data cleaning
    # a) knowing missing value
    # missing_value_info(df_train)
    # b) constant values
