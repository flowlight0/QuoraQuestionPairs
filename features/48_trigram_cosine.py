import os
import sys
import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
    print("Transform question1")
    X1 = vectorizer.transform(df.question1.values.astype(str))
    print("Transform question2")
    X2 = vectorizer.transform(df.question2.values.astype(str))

    values = []
    for i in tqdm(range(df.shape[0])):
        values.append(np.dot(X1[i], X2[i].T)[0, 0])

    df[column_name] = values
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = nltk_stemming_without_stopwords(dict(generate_filename_from_prefix(options.data_prefix))['train'])

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, norm='l2', ngram_range=(3, 3))
    vectorizer.fit_transform(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=vectorizer)


if __name__ == "__main__":
    main()
