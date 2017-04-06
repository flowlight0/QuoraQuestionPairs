import json

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from models.utils import common_model_parser, log_result


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = bst.get_fscore()
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def get_calibrator_file(model_file, config):
    calibrator_suffix = config['model']['calibrator_suffix']
    return model_file + '.' + calibrator_suffix


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    feature_ids = config['features']
    feature_columns = list(map(lambda s: 'f' + str(s), feature_ids))
    data = pd.read_csv(options.data_file)

    if options.train:
        X_train, X_valid, y_train, y_valid = \
            train_test_split(data[feature_columns], data['y'], test_size=0.2, random_state=4242)

        calibrator_file = get_calibrator_file(model_file=options.model_file, config=config)
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        params = config['model']['params']
        bst = xgb.train(params=params['booster'], dtrain=d_train, evals=watchlist, **params['train'])
        joblib.dump(bst, options.model_file)

        p_train = bst.predict(d_train)
        p_valid = bst.predict(d_valid)

        lr = LogisticRegression()
        lr.fit(p_train.reshape(-1, 1), y_train)
        p_calibrated = lr.predict_proba(p_valid.reshape(-1, 1))[:, 1]

        log_result(y_train, p_train, config, options.log_file)
        log_result(y_valid, p_valid, config, options.log_file)
        log_result(y_train, lr.predict_proba(p_train.reshape(-1, 1))[:, 1], config, options.log_file)
        log_result(y_valid, p_calibrated, config, options.log_file)

        add_feature_importance(bst, options.log_file)

        joblib.dump(lr, calibrator_file)
    else:
        bst = joblib.load(options.model_file)
        lr = joblib.load(get_calibrator_file(model_file=options.model_file, config=config))

        X = xgb.DMatrix(data[feature_columns])
        data['is_duplicate'] = lr.predict_proba(bst.predict(X).reshape(-1, 1))[:, 1]
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
