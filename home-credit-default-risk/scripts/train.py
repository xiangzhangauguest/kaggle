# encoding: utf-8

import logging

import numpy as np
import scipy
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb

logging.basicConfig(level=logging.DEBUG)


def train_xgb(params, features, labels):
    if not isinstance(features, scipy.sparse.csr.csr_matrix):
        # feature_matrix = feature_matrix.astype('float64', copy=False)
        feature_matrix = np.nan_to_num(features, copy=False)
    else:
        # feature_matrix = feature_matrix.astype('float64')
        pass

    dtrain = xgb.DMatrix(features, labels)

    xgb_model = xgb.train(params, dtrain, num_boost_round=800)
    # early_stopping_rounds=50)

    train_pred = xgb_model.predict(dtrain)
    if train_pred.shape[0] > 1:
        train_auc = roc_auc_score(labels, train_pred)
        print ("训练集的AUC为{0}".format(train_auc))

    print ("训练集log_loss=%s" % (log_loss(labels, train_pred)))

    return xgb_model