import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
from models.utils import common_model_parser, calculate_statistics


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = dict(zip(bst.feature_name(), bst.feature_importance().tolist()))
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def add_train_score(y_train, p_train, log_file: str, weight):
    log_data = json.load(open(log_file))
    log_data['results']['train_log_loss'] = log_loss(y_train, p_train, sample_weight=weight)
    print(log_data)


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    data = pd.read_csv(options.data_file)
    feature_columns = data.columns[data.columns != 'y']
    categorical_feature_columns = [feature for feature in feature_columns if feature.endswith('.cat')]
    print("Categorical features: {}".format(categorical_feature_columns), file=sys.stderr)

    if options.train:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=114514)
        negative_weight = (data.y.sum() / config['model']["target_positive_ratio"] - data.y.sum()) / (data.y == 0).sum()

        models = []
        stats = {"results": [], "config": config}
        data['prediction'] = [0] * data.shape[0]
        for train, valid in skf.split(data[feature_columns], data['y']):
            train_data = data.ix[train]
            valid_data = data.ix[valid]
            X_train, y_train = train_data[feature_columns], train_data['y']
            X_valid, y_valid = valid_data[feature_columns], valid_data['y']

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
            models.append(gbm)

            p_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
            p_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
            data.ix[valid]['prediction'] = p_valid
            stat = calculate_statistics(pred=p_valid, true=y_valid, weight=w_valid)
            stat['results']['train_log_loss'] = log_loss(y_train, p_train, sample_weight=w_train)
            stats["results"].append(stat["results"])
        joblib.dump(models, options.model_file)
        data[['prediction']].to_csv(options.model_file + '.train.pred', index=False)
        json.dump(stats, open(options.log_file, 'w'), sort_keys=True, indent=4)
    else:
        models = joblib.load(options.model_file)
        data['is_duplicate'] = np.zeros(data.shape[0])
        preds = np.zeros((data.shape[0], len(models)))
        for i, gbm in enumerate(models):
            preds[:, i] = gbm.predict(data[feature_columns])
        data['is_duplicate'] = preds.mean(axis=1)
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
