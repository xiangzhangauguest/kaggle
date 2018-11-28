import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import datetime

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
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

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y


if __name__ == "__main__":
    print("load train data")
    train_df = load_df("../input/train_v2.csv")
    print("load test data")
    test_df = load_df("../input/test_v2.csv")
    print("loading done.")

    train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
    gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]

    print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))
    cols_to_drop = const_cols
    # train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
    train_df = train_df.drop(cols_to_drop, axis=1)
    test_df = test_df.drop(cols_to_drop, axis=1)

    # Impute 0 for missing target values
    train_df["totals.transactionRevenue"].fillna(0, inplace=True)
    train_y = train_df["totals.transactionRevenue"].values
    train_id = train_df["fullVisitorId"].values
    test_id = test_df["fullVisitorId"].values

    # label encode the categorical variables and convert the numerical variables to float
    cat_cols = ["channelGrouping", "device.browser",
                "device.deviceCategory", "device.operatingSystem",
                "geoNetwork.city", "geoNetwork.continent",
                "geoNetwork.country", "geoNetwork.metro",
                "geoNetwork.networkDomain", "geoNetwork.region",
                "geoNetwork.subContinent", "trafficSource.adContent",
                "trafficSource.adwordsClickInfo.adNetworkType",
                "trafficSource.adwordsClickInfo.gclId",
                "trafficSource.adwordsClickInfo.page",
                "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
                "trafficSource.keyword", "trafficSource.medium",
                "trafficSource.referralPath", "trafficSource.source",
                'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
    for col in cat_cols:
        print(col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',
                'totals.newVisits']
    for col in num_cols:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)

    # Split the train dataset into development and valid based on time
    train_df['date'] = pd.to_datetime(train_df['date'], format="%Y%m%d")
    test_df['date'] = pd.to_datetime(test_df['date'], format="%Y%m%d")
    dev_df = train_df[train_df['date'] <= datetime.date(2017, 5, 31)]
    val_df = train_df[train_df['date'] > datetime.date(2017, 5, 31)]
    dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
    val_y = np.log1p(val_df["totals.transactionRevenue"].values)

    dev_X = dev_df[cat_cols + num_cols]
    val_X = val_df[cat_cols + num_cols]
    test_X = test_df[cat_cols + num_cols]

    # Training the model #
    pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    pred_val[pred_val < 0] = 0
    val_pred_df = pd.DataFrame({"fullVisitorId": val_df["fullVisitorId"].values})
    val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
    val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
    # print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
    print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values),
                                             np.log1p(val_pred_df["PredictedRevenue"].values))))

    sub_df = pd.DataFrame({"fullVisitorId": test_id})
    pred_test[pred_test < 0] = 0
    sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
    sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
    sub_df.to_csv("baseline_lgb.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=15)
    plt.show()