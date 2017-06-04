import pandas as pd

from ffmutil import FFMCV, create_ffm_file

fields = [
    '1grams_stem_q1',
    '1grams_stem_q2',
    '2grams_stem_q1',
    '2grams_stem_q2',
    '3grams_stem_q1',
    '3grams_stem_q2',
    '4grams_stem_q1',
    '4grams_stem_q2',
    '5grams_stem_q1',
    '5grams_stem_q2',
]


def main():
    ffmcv = FFMCV('ffm_001')
    preds = ffmcv.train(fields)
    train_pred = pd.DataFrame()
    train_pred['ffm_001'] = preds
    train_pred.to_csv('../data/working/20000_train.csv', index=False)

    test_ffm_filepath = create_ffm_file('test', fields, 'ffm_001')
    test_pred = pd.DataFrame()
    test_pred['ffm_001'] = ffmcv.predict(test_ffm_filepath)
    test_pred.to_csv('../data/working/20000_test.csv', index=False)

if __name__ == '__main__':
    main()
