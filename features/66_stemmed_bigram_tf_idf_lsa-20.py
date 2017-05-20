import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from features.transform import nltk_stemming
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_feature(data_file, vectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_stemming(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    X1 = vectorizer.transform(df.question1.values.astype(str))
    X2 = vectorizer.transform(df.question2.values.astype(str))
    X = np.hstack((X1, X2))

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
    df_train = nltk_stemming(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    df_test = nltk_stemming(dict(generate_filename_from_prefix(options.data_prefix))['test'])
    train_qs = pd.Series(df_train['question1'].tolist() +
                         df_train['question2'].tolist() +
                         df_test['question1'].tolist() +
                         df_test['question2'].tolist()).astype(str)

    pipeline = make_pipeline(
        TfidfVectorizer(max_df=0.5, min_df=2, norm='l2', ngram_range=(1, 2)),
        TruncatedSVD(n_components=20, n_iter=10)
    )
    pipeline.fit(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=pipeline)


if __name__ == "__main__":
    main()
