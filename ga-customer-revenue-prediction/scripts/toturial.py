import pandas as pd
import numpy as np
import json
import datetime as datetime
from datetime import timedelta, date
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta, date
import seaborn as sns
import matplotlib.cm as CM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import lightgbm as lgb

train_data = pd.read_csv("../input/train.csv", dtype=object)
print train_data.head()
print train_data.describe()
print train_data.columns.values

train_data.channelGrouping.value_counts().plot(kind="bar",title="channelGrouping distro",figsize=(8,8),rot=25,colormap='Paired')

train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")
train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')
print train_data.head(1)[["date","visitStartTime"]]

list_of_devices = train_data.device.apply(json.loads).tolist()
keys = set()
for devices_iter in list_of_devices:
    for list_element in list(devices_iter.keys()):
        keys.add(list_element)
print "keys existed in device attribute are:{}".format(keys)
tmp_device_df = pd.DataFrame(train_data.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]
print tmp_device_df.head()
print tmp_device_df.describe()

train_data["revenue"] = pd.DataFrame(train_data.totals.apply(json.loads).tolist())[["transactionRevenue"]]

df_train = train_data.drop(["date", "sessionId", "socialEngagementType", "visitStartTime", "visitId", "fullVisitorId" , "revenue"], axis=1)

devices_df = pd.DataFrame(df_train.device.apply(json.loads).tolist())[["browser", "operatingSystem", "deviceCategory", "isMobile"]]
geo_df = pd.DataFrame(df_train.geoNetwork.apply(json.loads).tolist())[["continent", "subContinent", "country", "city"]]
traffic_source_df = pd.DataFrame(df_train.trafficSource.apply(json.loads).tolist())[["keyword", "medium", "source"]]
totals_df = pd.DataFrame(df_train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]


df_train = pd.concat([df_train, devices_df, geo_df, traffic_source_df, totals_df], axis=1)
df_train = df_train.drop(["device", "geoNetwork", "trafficSource", "totals"], axis=1)
print df_train.head()
print df_train.describe()
print df_train.columns.values

df_train["transactionRevenue"] = df_train["transactionRevenue"].fillna(0)
df_train["bounces"] = df_train["bounces"].fillna(0)
df_train["pageviews"] = df_train["pageviews"].fillna(0)
df_train["hits"] = df_train["hits"].fillna(0)
df_train["newVisits"] = df_train["newVisits"].fillna(0)

df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

df_train["transactionRevenue"] = df_train["transactionRevenue"].astype(np.float)
df_test["transactionRevenue"] = df_test["transactionRevenue"].astype(np.float)
print "Finaly, we have these columns for our regression problems: {}".format(df_train.columns)

categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']

numerical_features = ['visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']


for column_iter in categorical_features:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[column_iter].values) + list(df_test[column_iter].values))
    df_train[column_iter] = lbl.transform(list(df_train[column_iter].values.astype('str')))
    df_test[column_iter] = lbl.transform(list(df_test[column_iter].values.astype('str')))

for column_iter in numerical_features:
    df_train[column_iter] = df_train[column_iter].astype(np.float)
    df_test[column_iter] = df_test[column_iter].astype(np.float)

params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
}
lgb_train = lgb.Dataset(df_train.loc[:,df_train.columns != "transactionRevenue"], np.log1p(df_train.loc[:,"transactionRevenue"]))
lgb_eval = lgb.Dataset(df_test.loc[:,df_test.columns != "transactionRevenue"], np.log1p(df_test.loc[:,"transactionRevenue"]), reference=lgb_train)
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)

predicted_revenue = gbm.predict(df_test.loc[:,df_test.columns != "transactionRevenue"], num_iteration=gbm.best_iteration)
predicted_revenue[predicted_revenue < 0] = 0
df_test["predicted"] = np.expm1(predicted_revenue)
df_test[["transactionRevenue","predicted"]].head(10)



