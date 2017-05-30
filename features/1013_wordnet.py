import os

import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from features.transform import nltk_tokenize
from features.utils import common_feature_parser
from features.utils import feature_output_file

lemmatizer = WordNetLemmatizer()


def calculate_max_wup(q1_synsets, q2_synsets):
    max_wup = -1
    for q1_syn in q1_synsets:
        for q2_syn in q2_synsets:
            try:
                wup = q1_syn.wup_similarity(q2_syn)
                if wup > max_wup:
                    max_wup = wup
            except:
                pass
    return max_wup


def calculate_max_path(q1_synsets, q2_synsets):
    max_path = -1
    for q1_syn in q1_synsets:
        for q2_syn in q2_synsets:
            try:
                path = q1_syn.path_similarity(q2_syn)
                if path > max_path:
                    max_path = path
            except:
                pass
    return max_path


def calculate_max_lch(q1_synsets, q2_synsets):
    max_lch = -1
    for q1_syn in q1_synsets:
        for q2_syn in q2_synsets:
            try:
                lch = q1_syn.lch_similarity(q2_syn)
                if lch > max_lch:
                    max_lch = lch
            except:
                pass
    return max_lch


def create_feature(q1, q2):
    q1_set = set(q1.split())
    q2_set = set(q2.split())

    q1_diff = list(q1_set - q2_set)
    q2_diff = list(q2_set - q1_set)

    result = {
        'q1_diff': len(q1_diff),
        'q2_diff': len(q2_diff)
    }

    if len(q1_diff) == 1 and len(q2_diff) == 1:
        q1_synsets = wordnet.synsets(lemmatizer.lemmatize(q1_diff[0]))
        q2_synsets = wordnet.synsets(lemmatizer.lemmatize(q2_diff[0]))
        if len(q1_synsets) > 0 and len(q2_synsets) > 0:
            result['max_wup_similarity'] = calculate_max_wup(q1_synsets, q2_synsets)
            result['max_lch_similarity'] = calculate_max_lch(q1_synsets, q2_synsets)
            result['max_path_similarity'] = calculate_max_path(q1_synsets, q2_synsets)
    return result


def create_features(data_path):
    print('data_path file: {}'.format(data_path))
    data = nltk_tokenize(data_path)#[:1000]

    features = Parallel(n_jobs=-1, verbose=5)(
        delayed(create_feature)(q1, q2)
        for q1, q2 in zip(data.question1.astype(str), data.question2.astype(str))
    )
    df = pd.DataFrame(features)
    df.to_csv(feature_output_file(data_path), index=False, float_format='%.5f')


def create_features_files(train_path, test_path):
    print(train_path, test_path)
    if os.path.exists(feature_output_file(train_path)) and os.path.exists(feature_output_file(test_path)):
        print('File exists {}.'.format(feature_output_file(train_path)) + ", " + feature_output_file(test_path))
        return

    print('Creating feature for train')
    create_features(train_path)

    print('Creating feature for test')
    create_features(test_path)


def main():
    options = common_feature_parser().parse_args()
    train_path = os.path.join(options.data_prefix, 'train.csv')
    test_path = os.path.join(options.data_prefix, 'test.csv')

    create_features_files(train_path, test_path)


if __name__ == '__main__':
    main()
