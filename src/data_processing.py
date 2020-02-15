import os
import numpy as np
import json  # to convert json in df
import pandas as pd
from pandas import json_normalize  # to normalize the json file
import random
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt

dir_path = "../data/"
p = 0.5 # fractional number to skip rows and read just a random sample of the our dataset.


from src.visualization import Visualization

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
        column_as_df.columns = [f"{column}{subcolumn}" for subcolumn in column_as_df.columns]
        column_as_df.columns = [subcolumn.replace('.', '') for subcolumn in column_as_df.columns]
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        print(list(column_as_df.columns.values))
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df


def missing_value_info(data):
    columns = data.columns[data.isnull().any()].tolist()
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending=False)
    # getting the percent and order of null
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    percentList = percent.tolist()
    percentLabelList = percent.index.tolist()

    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print("Total columns at least one Values: ")
    print(df[~(df['Total'] == 0)])  # Returning values of nulls different of 0

    print("\n Total of Sales % of Total: ", round((df_train[df_train['totalstransactionRevenue'] != np.nan][
         'totalstransactionRevenue'].count() / len(df_train['totalstransactionRevenue']) * 100), 4))

    tuple_data = [(i, j) for (i, j) in zip(percentList, percentLabelList) if i > 0]
    percentList, percentLabelList = [list(c) for c in zip(*tuple_data)]
    percentListSize = len(percentList)
    ind = np.arange(percentListSize)
    plt.bar(ind, percentList, width=0.3)
    plt.xticks(ind, percentLabelList, rotation=90)
    plt.show()

    return columns


def constant_value_info(train_df):
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]
    return const_cols


def drop_features(df):
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
    result = df.drop(to_drop, axis=1)
    if 'trafficSourcecampaignCode' in df.columns:
        result = df.drop(['trafficSourcecampaignCode'], axis=1)
    return result


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




if __name__ == '__main__':
    # 1. load data to df, and parse jason, then output to csv
    df_train = json_read("train.csv")
    df_test = json_read("test.csv")

    # process data feature
    # df_train['date'] = df_train['date'].apply(
    #     lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

    # missing values
    # missing_columns = missing_value_info(df_train)

    # constant values
    # const_columns = constant_value_info(df_train)

    # drop features
    df_train = drop_features(df_train)
    df_test = drop_features(df_test)

    # fill na
    df_train, df_test = change_feature_type_and_fill_na(df_train, df_test)

    # category to number
    df_train, df_test = category_to_number(df_train, df_test)
    print(df_train.info())
    print(df_test.info())

    # # add index name
    # df_train.rename(index={0: "index"})
    # df_test.rename(index={0: "index"})



    df_train.to_csv('../data/train_concise.csv', index=False)
    df_test.to_csv('../data/test_concise.csv', index=False)
