import os
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser
from features.transform import nltk_tokenize


all_counter = Counter()
dup_counter = Counter()


MIN_FREQ = 1000


def create_question_feature(words, prefix):
    words = [w for w in words if all_counter[w] >= MIN_FREQ]
    dup_probs = [dup_counter[w] / all_counter[w] for w in words]

    result = {
        'words_freq': len(words)
    }
    if len(words) > 0:
        result['ave_dup_prob'] = sum(dup_probs) / len(words)
        result['max_dup_prob'] = max(dup_probs)
        result['min_dup_prob'] = min(dup_probs)
    return {prefix + key: value for key, value in result.items()}


def create_feature(q1, q2):
    return {
        **create_question_feature(q1.split(), 'q1'),
        **create_question_feature(q2.split(), 'q2')
    }


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
        words1 = q1.split()
        words2 = q2.split()
        all_counter.update(words1)
        all_counter.update(words2)
        if dup:
            dup_counter.update(words1)
            dup_counter.update(words2)

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
