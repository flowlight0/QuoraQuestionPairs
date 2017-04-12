import json

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from models.utils import common_model_parser, log_result
from run import scale_dataset


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = bst.get_fscore()
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    data = pd.read_csv(options.data_file)
    feature_columns = data.columns[data.columns != 'y']

    if options.train:
        train, valid = train_test_split(data, test_size=0.2, random_state=334)
        train = scale_dataset(data=train, target_positive_ratio=0.191)
        valid = scale_dataset(data=valid, target_positive_ratio=0.191)
        X_train, y_train = train[feature_columns], train['y']
        X_valid, y_valid = valid[feature_columns], valid['y']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        params = config['model']['params']
        bst = xgb.train(params=params['booster'], dtrain=d_train, evals=watchlist, **params['train'])
        joblib.dump(bst, options.model_file)

        p_valid = bst.predict(d_valid)
        log_result(y_valid, p_valid, config, options.log_file)
        add_feature_importance(bst, options.log_file)
    else:
        bst = joblib.load(options.model_file)
        data['is_duplicate'] = bst.predict(xgb.DMatrix(data[feature_columns]))
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
