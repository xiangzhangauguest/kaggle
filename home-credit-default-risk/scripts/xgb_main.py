# encoding: utf-8

import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    try:
        import load_data
        import train
    except:
        logging.exception("Import error.")

    application_train_file = "../data/application_train.csv"
    params = {
        "objective": "binary:logistic",
        "alpha": 8,
        "lambda": 9,
        "colsample_bytree": 0.8,
        "gamma": 5,
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 100,
        'scale_pos_weight': 0.9,
        "subsample": 1,
        "nthread": 16,
        'silent': True,
        'eval_metric': ['logloss'],
        'verbose_eval': True,
        'seed': 2018
    }

    logging.info("Load application train data.")
    sk_id_curr, labels, features = load_data.load_application_train(application_train_file)

    logging.info("Train xgboost start.")
    xgb_model = train.train_xgb(params, features, labels)

