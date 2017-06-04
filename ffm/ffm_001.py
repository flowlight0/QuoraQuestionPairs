import pandas as pd

from ffmutil import FFMCV

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
    train_pred.to_csv('../data/working/10000_train.csv', index=False)

if __name__ == '__main__':
    main()
