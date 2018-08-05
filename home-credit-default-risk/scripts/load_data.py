# encoding: utf-8

import datetime
import logging

import numpy as np
import scipy

logging.basicConfig(level=logging.DEBUG)


def load_application_train(data_file):
    logging.info('load data begin')
    begin = datetime.datetime.now()
    matrix = np.loadtxt(data_file, str, delimiter='\t')
    end = datetime.datetime.now()
    logging.info('load data end,耗时:%s' % (end - begin))

    sk_id_curr = matrix[:, 0:1]
    labels = matrix[:, 1:2]
    features = matrix[:, 2:]

    # feature_label = feature_label.astype('float64', copy=False)

    logging.info("features shape: ")
    logging.info(features.shape)
    labels = np.int_(labels)
    return sk_id_curr, labels, features
