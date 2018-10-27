# encoding: utf-8

import datetime
import logging

import numpy as np
import scipy
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def load_application_train(data_file):
    try:
        logging.info('load data begin')
        begin = datetime.datetime.now()
        matrix = pd.read_csv(data_file)
        end = datetime.datetime.now()
        logging.info('load data end,耗时:%s' % (end - begin))

        logging.debug(matrix)
        logging.debug("Matrix shape: {0}".format(matrix.shape))
        sk_id_curr = matrix.iloc[:, 0:1]
        labels = matrix.iloc[:, 1:2]
        features = matrix.iloc[:, 2:]

        feat_names = [f for f in matrix.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

        # feature_label = feature_label.astype('float64', copy=False)

        logging.info("features shape: {0}".format(features.shape))
        # labels = np.int_(labels)
        return sk_id_curr, labels, features
    except Exception as e:
        logging.exception("Load error.")