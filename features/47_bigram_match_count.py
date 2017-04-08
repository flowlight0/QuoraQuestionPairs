import os
import sys

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from features.transform import nltk_stemming, nltk_stemming_without_stopwords
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def calc_feature(row, vectorizer):
    q1 = vectorizer.transform([str(row['question1'])])
    q2 = vectorizer.transform([str(row['question2'])])
    return (q1 * q2.T)[0, 0]


def create_feature(data_file, vectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_stemming(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])

    values = []
    for i in tqdm(range(df.shape[0])):
        q1 = vectorizer.transform([str(df.question1.values[i])])
        q2 = vectorizer.transform([str(df.question2.values[i])])
        values.append(np.dot(q1, q2.T)[0, 0])

    df[column_name] = values
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = nltk_stemming_without_stopwords(dict(generate_filename_from_prefix(options.data_prefix))['train'])

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    vectorizer = CountVectorizer(max_df=0.5, min_df=2, ngram_range=(2, 2), binary=True)
    vectorizer.fit_transform(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=vectorizer)


if __name__ == "__main__":
    main()
