import numpy as np
import json  # to convert json in df
import pandas as pd
from pandas import json_normalize  # to normalize the json file
import random
import matplotlib.pyplot as plt

DATA_PATH = '../data/'
PROCESS_DATA_PERCENT = 1  # fractional number to skip rows and read just a random sample of the our dataset.
JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']  # Columns that have json format


# Transform the json format columns in table
def json_read(df):
    # joining the [ path + df received]
    data_frame = DATA_PATH + df

    df = pd.read_csv(data_frame,
                     converters={column: json.loads for column in JSON_COLS},  # loading the json columns properly
                     dtype={'fullVisitorId': 'str'},  # transforming this column to string
                     skiprows=lambda i: i > 0 and random.random() > PROCESS_DATA_PERCENT,
                     )  # Number of rows that will be imported randomly
    return df


def process_json_data(df):
    for column in JSON_COLS:
        # It will normalize and set the json to a table
        column_as_df = json_normalize(df[column])
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f'{column}{subcolumn}' for subcolumn in column_as_df.columns]
        column_as_df.columns = [subcolumn.replace('.', '') for subcolumn in column_as_df.columns]
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1, errors='ignore').merge(column_as_df, right_index=True, left_index=True)
        # print(list(column_as_df.columns.values))
    # print(f'Shape: {df.shape}')
    return df


def missing_value_info(data):
    columns = data.columns[data.isnull().any()].tolist()
    # getting the sum of null values and ordering
    total = data.isnull().sum().sort_values(ascending=False)
    # getting the percent and order of null
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    percent_list = percent.tolist()
    percent_label_list = percent.index.tolist()

    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    tuple_data = [(i, j) for (i, j) in zip(percent_list, percent_label_list) if i > 0]
    percent_list, percent_label_list = [list(c) for c in zip(*tuple_data)]
    percent_list_size = len(percent_list)
    ind = np.arange(percent_list_size)
    plt.bar(ind, percent_list, width=0.3)
    plt.xticks(ind, percent_label_list, rotation=90)
    plt.show()

    return columns


def constant_value_info(train_df):
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]
    return const_cols


def drop_features(df):
    to_drop = ['sessionId', 'socialEngagementType', 'devicebrowserVersion', 'devicebrowserSize', 'deviceflashVersion',
               'devicelanguage', "visitId",
               'devicemobileDeviceBranding', 'devicemobileDeviceInfo', 'devicemobileDeviceMarketingName',
               'devicemobileDeviceModel',
               'devicemobileInputSelector', 'deviceoperatingSystemVersion', 'devicescreenColors',
               'devicescreenResolution',
               'geoNetworkcityId', 'geoNetworklatitude', 'geoNetworklongitude', 'geoNetworknetworkLocation',
               'trafficSourceadwordsClickInfocriteriaParameters', 'trafficSourceadwordsClickInfogclId',
               'trafficSourcecampaign',
               'trafficSourceadwordsClickInfopage',
               'trafficSourceadwordsClickInfoslot',
               'trafficSourceadContent', 'trafficSourceadwordsClickInfoadNetworkType',
               'totalsbounces', 'totalsnewVisits', 'totalsvisits',
               'trafficSourceisTrueDirect',
               'trafficSourceadwordsClickInfoisVideoAd']
    df.drop(to_drop, axis=1, errors='ignore', inplace=True)
    if 'trafficSourcecampaignCode' in df.columns:
        df.drop(['trafficSourcecampaignCode'], axis=1, errors='ignore', inplace=True)
    return df


def change_feature_type_and_fill_na(df):
    if 'geoNetworkcity' in df.columns:
        df.loc[df['geoNetworkcity'] == '(not set)', 'geoNetworkcity'] = np.nan
        df.loc[df['geoNetworkcity'] == 'not available in demo dataset', 'geoNetworkcity'] = np.nan
        df['geoNetworkcity'].fillna('NaN', inplace=True)

    if 'geoNetworknetworkDomain' in df.columns:
        df.loc[df['geoNetworknetworkDomain'] == '(not set)', 'geoNetworknetworkDomain'] = np.nan
        df.loc[df['geoNetworknetworkDomain'] == 'not available in demo dataset', 'geoNetworknetworkDomain'] = np.nan
        df['geoNetworknetworkDomain'].fillna('NaN', inplace=True)

    if 'geoNetworkmetro' in df.columns:
        df.loc[df['geoNetworkmetro'] == '(not set)', 'geoNetworkmetro'] = np.nan
        df.loc[df['geoNetworkmetro'] == 'not available in demo dataset', 'geoNetworkmetro'] = np.nan
        df['geoNetworkmetro'].fillna('NaN', inplace=True)

    if 'geoNetworknetworkDomain' in df.columns:
        df.loc[df['geoNetworknetworkDomain'] == 'not available in demo dataset', 'geoNetworknetworkDomain'] = np.nan
        df['geoNetworknetworkDomain'].fillna('NaN', inplace=True)

    if 'geoNetworkregion' in df.columns:
        df.loc[df['geoNetworkregion'] == 'not available in demo dataset', 'geoNetworkregion'] = np.nan
        df['geoNetworkregion'].fillna('NaN', inplace=True)

    if 'trafficSourcekeyword' in df.columns:
        df.loc[df['trafficSourcekeyword'] == '(not provided)', 'trafficSourcekeyword'] = np.nan
        df.loc[df['trafficSourcekeyword'] == '(not set)', 'trafficSourcekeyword'] = np.nan
        df['trafficSourcekeyword'].fillna('NaN', inplace=True)

    if 'totalshits' in df.columns:
        df['totalshits'] = df['totalshits'].astype(int)  # setting numerical to int

    if 'totalspageviews' in df.columns:
        df['totalspageviews'].fillna(1, inplace=True)  # filling NA's with 1
        df['totalspageviews'] = df['totalspageviews'].astype(int)  # setting numerical column as integer

    if 'totalstransactionRevenue' in df.columns:
        df['totalstransactionRevenue'] = df['totalstransactionRevenue'].fillna(0.0).astype(float)



    return df


def process_data(raw_data):
    print('raw_data shape ' + str(raw_data.shape))
    data_processed_json = process_json_data(raw_data)
    print('data_processed_json shape ' + str(data_processed_json.shape))
    data_dropped_feature = drop_features(data_processed_json)
    print('data_dropped_feature shape ' + str(data_dropped_feature.shape))
    data_filled_na = change_feature_type_and_fill_na(data_dropped_feature)
    print('data_filled_na shape ' + str(data_filled_na.shape))
    return data_filled_na


if __name__ == '__main__':
    data_train_raw = json_read('train.csv')
    processed_train_data = process_data(data_train_raw)
    print(processed_train_data.info())
    processed_train_data.to_csv('../data/train_peng.csv', index=False)
    del data_train_raw, processed_train_data

    data_test_raw = json_read('test.csv')
    processed_test_data = process_data(data_test_raw)
    print(processed_test_data.info())
    processed_test_data.to_csv('../data/test_peng.csv', index=False)
    del data_test_raw, processed_test_data
