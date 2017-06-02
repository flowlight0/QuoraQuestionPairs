import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from features.tfkld import TfKLdVectorizer
from features.transform import nltk_tokenize, nltk_stemming_without_stopwords
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_feature(data_file, vectorizer, pipeline):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_tokenize(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    X1 = pipeline.transform(vectorizer.transform(df.question1.values.astype(str)))
    X2 = pipeline.transform(vectorizer.transform(df.question2.values.astype(str)))
    X = np.hstack((X1, X2))

    column_names = []
    for i in tqdm(range(X.shape[1])):
        column_name = column_name_prefix + '.' + str(i)
        df[column_name] = X[:, i]
        column_names.append(column_name)
    column_names = pd.Index(column_names)
    df[column_names] = X
    df[column_names].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = nltk_stemming_without_stopwords(dict(generate_filename_from_prefix(options.data_prefix))['train'])

    vectorizer = TfKLdVectorizer(alpha=0.5, divergence='kl', ngram_range=(1, 1), max_df=1.0, min_df=500)
    train_q1s = pd.Series(df_train['question1'].tolist())
    train_q2s = pd.Series(df_train['question2'].tolist())
    train_ys = pd.Series(df_train['is_duplicate'].tolist())
    vectorizer.fit(train_q1s, train_q2s, train_ys)

    train_qs = pd.Series(train_q1s.tolist() + train_q2s.tolist()).astype(str)
    value_qs = vectorizer.transform(train_qs)
    print(value_qs)

    pipeline = make_pipeline(
        TruncatedSVD(n_components=10)
    )
    pipeline.fit(value_qs)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=vectorizer, pipeline=pipeline)


if __name__ == "__main__":
    main()
