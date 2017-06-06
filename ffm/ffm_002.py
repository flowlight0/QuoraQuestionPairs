import os

import pandas as pd

from ffmutil import FFMCV, create_ffm_file


STORE_FEATURES_DIR = 'ffm_features'


def list_fields(original_features):
    feature_files = os.listdir(STORE_FEATURES_DIR)
    cand_fields = [f[len('test_'):] for f in feature_files if f.startswith('test_f')]
    return [
        field for field in cand_fields
        if int(field.split('_')[0][len('f'):]) in original_features
    ]

original_features = [
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 21, 22, 28, 30,
    31, 32, 44, 45, 46, 47, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 91, 92, 93, 109, 110, 111, 112,
    117, 118, 119, 145, 146, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1013, 1018, 1026, 1027, 1028, 1029, 1030
]


def main():
    ffm_name = 'ffm_002'

    fields = list_fields(original_features)

    ffmcv = FFMCV(ffm_name)
    preds = ffmcv.train(fields)
    train_pred = pd.DataFrame()
    train_pred[ffm_name] = preds
    train_pred.to_csv('../data/working/20001_train.csv', index=False)

    test_ffm_filepath = create_ffm_file('test', fields, ffm_name)
    test_pred = pd.DataFrame()
    test_pred[ffm_name] = ffmcv.predict(test_ffm_filepath)
    test_pred.to_csv('../data/working/20001_test.csv', index=False)

if __name__ == '__main__':
    main()
