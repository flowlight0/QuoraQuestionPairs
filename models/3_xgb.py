import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from models.utils import common_model_parser, log_result
from run import scale_dataset


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = bst.get_fscore()
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def add_train_score(y_train, p_train, log_file: str, weight):
    log_data = json.load(open(log_file))
    log_data['results']['train_log_loss'] = log_loss(y_train, p_train, sample_weight=weight)
    print(log_data)
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    data = pd.read_csv(options.data_file)
    feature_columns = data.columns[data.columns != 'y']

    if options.train:
        train, valid = train_test_split(data, test_size=0.2, random_state=334, stratify=data.y)
        X_train, y_train = train[feature_columns], train['y']
        X_valid, y_valid = valid[feature_columns], valid['y']

        negative_weight = (data.y.sum() / config['model']["target_positive_ratio"] - data.y.sum()) / (data.y == 0).sum()
        w_train = np.ones(X_train.shape[0])
        w_train[y_train == 0] *= negative_weight
        w_valid = np.ones(X_valid.shape[0])
        w_valid[y_valid == 0] *= negative_weight

        d_train = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        params = config['model']['params']
        bst = xgb.train(params=params['booster'], dtrain=d_train, evals=watchlist, **params['train'])
        joblib.dump(bst, options.model_file)

        p_train = bst.predict(d_train)
        p_valid = bst.predict(d_valid)
        log_result(y_valid, p_valid, config, options.log_file, weight=w_valid)
        add_feature_importance(bst, options.log_file)
        add_train_score(y_train=y_train, p_train=p_train, log_file=options.log_file, weight=w_train)
    else:
        bst = joblib.load(options.model_file)
        data['is_duplicate'] = bst.predict(xgb.DMatrix(data[feature_columns]))
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
