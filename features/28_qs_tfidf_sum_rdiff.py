import os

import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features.transform import nltk_stemming
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def calc_feature(row, vectorizer):
    m1 = np.sum(vectorizer.transform([str(row['question1'])]).data)
    m2 = np.sum(vectorizer.transform([str(row['question2'])]).data)
    if max(m1, m2) > 0:
        return abs(m1 - m2) / max(m1, m2)
    else:
        return 0


def create_feature(data_file, vectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_stemming(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(calc_feature, axis=1, raw=True, vectorizer=vectorizer)
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()

    df_train = nltk_stemming(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, norm='l2')
    vectorizer.fit_transform(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=vectorizer)

if __name__ == "__main__":
    main()
