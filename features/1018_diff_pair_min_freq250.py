import os
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser
from features.transform import nltk_tokenize


def list_diff_pairs(q1, q2):
    set_q1 = set(q1.split(" "))
    set_q2 = set(q2.split(" "))
    diff_q1 = set_q1 - set_q2
    diff_q2 = set_q2 - set_q1
    return [(word1, word2) for word1 in diff_q1 for word2 in diff_q2]


all_counter = Counter()
dup_counter = Counter()


MIN_FREQ = 250


def create_feature(q1, q2):
    set_q1 = set(q1.split(" "))
    set_q2 = set(q2.split(" "))
    diff_q1 = set_q1 - set_q2
    diff_q2 = set_q2 - set_q1
    diff_paris = [(word1, word2) for word1 in diff_q1 for word2 in diff_q2]
    diff_paris = [pair for pair in diff_paris if all_counter[pair] >= MIN_FREQ]

    dup_probs = [dup_counter[pair] / all_counter[pair] for pair in diff_paris]

    result = {
        'diff_q1': len(diff_q1),
        'diff_q2': len(diff_q2),
        'diff_pairs': len(diff_paris),
    }

    if len(dup_probs) > 0:
        result['ave_dup_prob'] = sum(dup_probs) / len(dup_probs)

    return result


def create_features(data_path):
    data = nltk_tokenize(data_path)

    feature_dicts = Parallel(n_jobs=-1, verbose=3)(
        delayed(create_feature)(q1, q2) for q1, q2 in zip(data.question1.astype(str), data.question2.astype(str))
    )

    df = pd.DataFrame(feature_dicts)
    df.to_csv(feature_output_file(data_path), index=False, float_format='%.5f')


def create_features_files(train_path, test_path):
    print(train_path, test_path)
    if os.path.exists(feature_output_file(train_path)) and os.path.exists(feature_output_file(test_path)):
        print('File exists {}.'.format(feature_output_file(train_path)) + ", " + feature_output_file(test_path))
        return

    print('Preprocessing')
    train = nltk_tokenize(train_path)
    for q1, q2, dup in tqdm(zip(train.question1.astype(str), train.question2.astype(str), train.is_duplicate)):
        diff_pairs = list_diff_pairs(q1, q2)
        all_counter.update(diff_pairs)
        if dup:
            dup_counter.update(diff_pairs)

    print('Creating feature for train')
    create_features(train_path)

    print('Creating feature for test')
    create_features(test_path)


def main():
    options = common_feature_parser().parse_args()
    train_path = os.path.join(options.data_prefix, 'train.csv')
    test_path = os.path.join(options.data_prefix, 'test.csv')

    create_features_files(train_path, test_path)

if __name__ == "__main__":
    main()
