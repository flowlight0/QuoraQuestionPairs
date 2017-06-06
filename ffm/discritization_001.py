import os
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ORIGINAL_FEATURES_DIR = '../data/working'
STORE_FEATURES_DIR = 'ffm_features'


def quantile(cat):
    bins = 128
    while bins > 8:
        try:
            return pd.qcut(cat, bins, labels=False)
        except:
            bins //= 2
    return None


def discretize(train, test):
    if train.dtype == np.int:
        return train, test

    cat = pd.concat([train, test])

    nan = np.isnan(cat)
    pos_inf = np.isinf(cat)
    neg_inf = np.isneginf(cat)

    cat[nan | pos_inf | neg_inf] = cat[~(nan | pos_inf | neg_inf)].mean()

    disc = quantile(cat)
    if disc is None:
        disc = pd.cut(cat, 32, labels=False)

    max_disc = disc.max()
    disc[nan] = max_disc + 1
    disc[pos_inf] = max_disc + 2
    disc[neg_inf] = max_disc + 3

    return disc[:len(train)], disc[len(train):]


def store_discretization_feature(feature_list, filename):
    with open(os.path.join(STORE_FEATURES_DIR, filename), 'wb') as f:
        pickle.dump(feature_list, f)


def create_discretization_feature(feature_i):
    train_path = os.path.join(ORIGINAL_FEATURES_DIR, '{}_train.csv'.format(feature_i))
    test_path = os.path.join(ORIGINAL_FEATURES_DIR, '{}_test.csv'.format(feature_i))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_name_prefix = 'f{}'.format(feature_i)
    for column in train.columns:
        train_f, test_f = train[column], test[column]
        train_q, test_q = discretize(train_f, test_f)

        train_list = [[i] for i in train_q.tolist()]
        test_list = [[i] for i in test_q.tolist()]

        train_q_path = 'train_{}_{}'.format(feature_name_prefix, column)
        test_q_path = 'test_{}_{}'.format(feature_name_prefix, column)
        store_discretization_feature(train_list, train_q_path)
        store_discretization_feature(test_list, test_q_path)


original_features = [
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 21, 22, 28, 30,
    31, 32, 44, 45, 46, 47, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 91, 92, 93, 109, 110, 111, 112,
    117, 118, 119, 145, 146, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1013, 1018, 1026, 1027, 1028, 1029, 1030
]


def main():
    Parallel(n_jobs=-1, verbose=5)(
        delayed(create_discretization_feature)(feature_i) for feature_i in original_features
    )

if __name__ == '__main__':
    main()