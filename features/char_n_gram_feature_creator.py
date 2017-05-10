import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.utils import get_stop_words, feature_output_file


class CharNGramSimilarityFeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options, max_df=0.5, min_df=20, ngram_range=(1, 1), binary=True):
        super().__init__(options)
        self.vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range, binary=binary,
                                          analyzer='char')
        self.stop_words = get_stop_words()
        self.read_func = pd.read_csv

    @staticmethod
    def get_num_rows(data):
        return data.shape[0]

    def calculate_row_feature(self, row: scipy.sparse.csr.csr_matrix):
        return np.linalg.norm(row.data)

    def get_column_name(self, input_file):
        prefix = 'f{0}'.format(os.path.basename(feature_output_file(input_file)).split('_')[0])
        return "{},{},{}".format(prefix + ".mc", prefix + ".jc", prefix + "ds")

    def write_feature(self, column_name, output_file, values):
        output_file = open(output_file, mode='w')
        print(column_name, file=output_file)
        for value in values:
            print(value, file=output_file)

    def calculate_features(self, data: Tuple[scipy.sparse.csr.csr_matrix, scipy.sparse.csr.csr_matrix]):
        X1, X2 = data
        values = []
        for i in tqdm(range(X1.shape[0])):
            s1 = set(X1.indices[X1.indptr[i]:X1.indptr[i + 1]])
            s2 = set(X2.indices[X2.indptr[i]:X2.indptr[i + 1]])
            inter = s1.intersection(s2)
            union = s1.union(s2)
            if len(union) > 0:
                jaccard = float(len(inter)) / len(union)
                dice = 2 * float(len(inter)) / (len(s1) + len(s2))
            else:
                jaccard = 0
                dice = 0
            values.append("{},{:.4f},{:.4f}".format(len(inter), jaccard, dice))
        return values

    def prepare(self):
        df_train = self.read_func(self.input_files['train'])
        df_test = self.read_func(self.input_files['test'])
        train_qs = pd.Series(df_train['question1'].fillna("").map(lambda x: x.lower()).tolist() +
                             df_train['question2'].fillna("").map(lambda x: x.lower()).tolist() +
                             df_test['question1'].fillna("").map(lambda x: x.lower()).tolist() +
                             df_test['question2'].fillna("").map(lambda x: x.lower()).tolist()).astype(str)
        self.vectorizer.fit(train_qs.values)

    def read_data(self, data_file):
        X1 = self.vectorizer.transform(
            self.read_func(data_file)['question1'].fillna("").map(lambda x: x.lower()).tolist())
        X2 = self.vectorizer.transform(
            self.read_func(data_file)['question2'].fillna("").map(lambda x: x.lower()).tolist())
        return X1, X2
