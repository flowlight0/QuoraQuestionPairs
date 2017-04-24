import os
import sys

import gensim
import pandas as pd
import numpy as np
from tqdm import tqdm

from features.transform import nltk_tokenize
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def calc_document_vector(docs, model: gensim.models.Doc2Vec):
    X = np.zeros((len(docs), 300))
    for i, doc in tqdm(enumerate(docs)):
        X[i] = model.infer_vector(doc.split(" "))
    return X


def create_word_match_feature(data_file, model: gensim.models.Doc2Vec):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_tokenize(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])

    X1 = calc_document_vector(df.question1.values.astype(str).tolist(), model)
    X2 = calc_document_vector(df.question2.values.astype(str).tolist(), model)
    X = np.hstack((X1, X2))

    column_names = []
    for i in tqdm(range(X.shape[1])):
        column_name = column_name_prefix + '.' + str(i)
        df[column_name] = X[:, i]
        column_names.append(column_name)
    column_names = pd.Index(column_names)
    print('Start to write dataset')
    df[column_names].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    m = gensim.models.Doc2Vec.load('../data/input/enwiki_dbow/doc2vec.bin')
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_word_match_feature(data_file=file_name, model=m)


if __name__ == "__main__":
    main()
