import json

import joblib
import pandas as pd
import numpy as np
import sys

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from models.utils import common_model_parser, log_result


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = dict(zip(bst.feature_name(), bst.feature_importance().tolist()))
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
    categorical_feature_columns = [feature for feature in feature_columns if feature.endswith('.cat')]
    print("Categorical features: {}".format(categorical_feature_columns), file=sys.stderr)

    if options.train:
        train, valid = train_test_split(data, test_size=0.2, random_state=334, stratify=data.y)
        X_train, y_train = train[feature_columns], train['y']
        X_valid, y_valid = valid[feature_columns], valid['y']

        negative_weight = (data.y.sum() / config['model']["target_positive_ratio"] - data.y.sum()) / (data.y == 0).sum()
        w_train = np.ones(X_train.shape[0])
        w_train[y_train == 0] *= negative_weight
        w_valid = np.ones(X_valid.shape[0])
        w_valid[y_valid == 0] *= negative_weight

        d_train = lgb.Dataset(data=X_train, label=y_train, weight=w_train,
                              categorical_feature=categorical_feature_columns)
        d_valid = lgb.Dataset(data=X_valid, label=y_valid, weight=w_valid,
                              categorical_feature=categorical_feature_columns)
        params = config['model']['params']
        gbm = lgb.train(params['booster'], d_train, valid_sets=d_valid, **params['train'])
        joblib.dump(gbm, options.model_file)

        p_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
        p_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

        log_result(y_valid, p_valid, config, options.log_file, weight=w_valid)
        add_feature_importance(gbm, options.log_file)
        add_train_score(y_train=y_train, p_train=p_train, log_file=options.log_file, weight=w_train)
    else:
        gbm = joblib.load(options.model_file)
        data['is_duplicate'] = gbm.predict(data[feature_columns])
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
