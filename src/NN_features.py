import os
import pandas as pd
import numpy as np
import datetime
import setuptools
import time
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, preprocessing, metrics


df_train = pd.read_csv("../data/train_concise.csv")
print(df_train.info())

df_train['totalstransactionRevenue'] = np.log1p(df_train['totalstransactionRevenue'])

df_train['date'] = pd.to_datetime(
    df_train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['weekday'] = df_train['date'].dt.weekday
df_train['weekofyear'] = df_train['date'].dt.weekofyear


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

# encode category to num
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))
    max_values[col] = df_train[col].max()


def separate_data(train):
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

    # Split the train dataset into development and valid based on time
    train['date'] = train['date'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:10])))

    dev_df = train[train['date'] <= datetime.date(2017, 5, 31)]
    val_df = train[train['date'] > datetime.date(2017, 5, 31)]
    dev_y = np.log1p(dev_df["totalstransactionRevenue"].values)
    val_y = np.log1p(val_df["totalstransactionRevenue"].values)

    dev_X = dev_df[cat_cols + num_cols]
    val_X = val_df[cat_cols + num_cols]

    print('==========final data==========')
    print('train_X shape ' + str(dev_X.shape))
    print('train_y shape ' + str(dev_y.shape))
    print('val_X shape ' + str(val_X.shape))
    print('val_y shape ' + str(val_y.shape))

    return dev_X, dev_y, val_X, val_y, dev_df, val_df


train_X, train_y, val_X, val_y, dev_df, val_df = separate_data(df_train)


# converting data to format which can be used by Keras
def get_keras_data(df, num_cols, cat_cols):
    cols = num_cols + cat_cols
    X = {col: np.array(df[col]) for col in cols}
    print("Data ready for Vectorization")
    return X

inp_dim = train_X.shape[1]
# train_X = get_keras_data(train_X, num_cols, cat_cols)
# val_X = get_keras_data(val_X, num_cols, cat_cols)
train_X = np.array(train_X)
train_y = np.array(train_y)

model = keras.models.Sequential()
model.add(keras.layers.Dense(50, input_dim=inp_dim, activation='relu'))
model.add(keras.layers.Dense(units=50, activation='relu'))
model.add(keras.layers.Dense(units=20, activation='relu'))
# model.add(keras.layers.GlobalMaxPool1D())
model.add(keras.layers.Dense(1, activation='relu'))

# model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

hist = model.fit(train_X, train_y, batch_size=100, epochs=10, validation_data=(val_X, val_y))


pred_y = model.predict(val_X, batch_size=100, verbose=1)


def validate(val_df, pred_val):
    val_pred_df = pd.DataFrame({"fullVisitorId": val_df["fullVisitorId"].values})
    val_pred_df["transactionRevenue"] = val_df["totalstransactionRevenue"].values
    val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
    val_pred_df = val_pred_df.groupby('fullVisitorId')[['transactionRevenue', 'PredictedRevenue']].sum().reset_index()
    print(np.sqrt(metrics.mean_squared_error(val_pred_df['transactionRevenue'].values,
                                             val_pred_df['PredictedRevenue'].values)))


pred_y[pred_y < 0] = 0
validate(val_df, pred_y)