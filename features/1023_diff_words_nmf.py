import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline

from tqdm import tqdm

from features.transform import nltk_tokenize
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def get_diff_words(q1, q2):
    set_q1 = set(q1.split(" "))
    set_q2 = set(q2.split(" "))
    diff_q1 = set_q1 - set_q2
    return diff_q1


def create_feature(data_file, vectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_tokenize(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    diff_words_q1 = [get_diff_words(q1, q2)
                        for q1, q2 in tqdm(zip(df.question1.astype(str), df.question1.astype(str)))]
    diff_words_q2 = [get_diff_words(q2, q1)
                        for q1, q2 in tqdm(zip(df.question1.astype(str), df.question1.astype(str)))]

    X = np.hstack((vectorizer.transform(diff_words_q1), vectorizer.transform(diff_words_q2)))

    column_names = []
    for i in range(X.shape[1]):
        column_name = column_name_prefix + '.' + str(i)
        df[column_name] = X[:, i]
        column_names.append(column_name)
    column_names = pd.Index(column_names)
    df[column_names] = X
    df[column_names].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = nltk_tokenize(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    df_test = nltk_tokenize(dict(generate_filename_from_prefix(options.data_prefix))['test'])

    train_diff_words_q1 = [get_diff_words(q1, q2) for q1, q2 in
                           tqdm(zip(df_train.question1.astype(str), df_train.question2.astype(str)))]
    train_diff_words_q2 = [get_diff_words(q2, q1) for q1, q2 in
                           tqdm(zip(df_train.question1.astype(str), df_train.question2.astype(str)))]
    test_diff_words_q1 = [get_diff_words(q1, q2) for q1, q2 in
                           tqdm(zip(df_test.question1.astype(str), df_test.question2.astype(str)))]
    test_diff_words_q2 = [get_diff_words(q2, q1) for q1, q2 in
                           tqdm(zip(df_test.question1.astype(str), df_test.question2.astype(str)))]

    pipeline = make_pipeline(
        CountVectorizer(min_df=4, tokenizer=lambda a: a, lowercase=False),
        NMF(n_components=10, random_state=1, l1_ratio=.15, verbose=True)
    )
    pipeline.fit(train_diff_words_q1 + train_diff_words_q2 + test_diff_words_q1 + test_diff_words_q2)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=pipeline)


if __name__ == "__main__":
    main()
