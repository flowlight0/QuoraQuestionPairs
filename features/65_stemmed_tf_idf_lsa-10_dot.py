import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from features.transform import nltk_stemming
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_feature(data_file, vectorizer):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_stemming(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    X1 = vectorizer.transform(df.question1.values.astype(str))
    X2 = vectorizer.transform(df.question2.values.astype(str))
    values = []
    for i in tqdm(range(X1.shape[0])):
        values.append(np.dot(X1[i], X2[i].T))
    df[column_name] = values
    df[column_name].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = nltk_stemming(dict(generate_filename_from_prefix(options.data_prefix))['train'])

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

    pipeline = make_pipeline(
        TfidfVectorizer(max_df=0.5, min_df=2, norm='l2', ngram_range=(1, 2)),
        TruncatedSVD(n_components=10)
    )
    pipeline.fit(train_qs.values)

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, vectorizer=pipeline)


if __name__ == "__main__":
    main()
