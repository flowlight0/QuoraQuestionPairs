import json

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from models.utils import common_model_parser, log_result


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = bst.get_fscore()
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    feature_ids = config['features']
    feature_columns = list(map(lambda s: 'f' + str(s), feature_ids))
    data = pd.read_csv(options.data_file)

    if options.train:
        X_train, X_valid, y_train, y_valid = \
            train_test_split(data[feature_columns], data['y'], test_size=0.2, random_state=4242)

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
