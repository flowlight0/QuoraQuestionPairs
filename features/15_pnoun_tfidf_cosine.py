import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features.transform import nltk_pos_tag
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def convert(df: pd.DataFrame):
    df['question1'] = df['question1'].apply(
        lambda stags: " ".join([s for (s, tag) in ast.literal_eval(stags) if tag.startswith('NNP')])
    )
    df['question2'] = df['question2'].apply(
        lambda stags: " ".join([s for (s, tag) in ast.literal_eval(stags) if tag.startswith('NNP')])
    )
    return df


def create_feature(data_file, vectorizer: TfidfVectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = convert(nltk_pos_tag(data_file))

    cosine_values = []
    q1vec = vectorizer.transform(df['question1'].apply(lambda x: x if type(x) == str else '').values)
    q2vec = vectorizer.transform(df['question2'].apply(lambda x: x if type(x) == str else '').values)
    for i in range(df.shape[0]):
        cosine_values.append(round(float(np.dot(q1vec[i], q2vec[i].T)[0, 0]), 5))
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = cosine_values
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    print(sys.argv[0], file=sys.stderr)
    df_train = convert(nltk_pos_tag(dict(generate_filename_from_prefix(options.data_prefix))['train']))
    print(df_train.head(), file=sys.stderr)

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, norm='l2')
    vectorizer.fit_transform(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=vectorizer)


if __name__ == "__main__":
    main()
