from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_stemming_without_stopwords
from features.utils import get_stop_words, common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=8, ngram_range=(4, 4))
        self.stop_words = get_stop_words()

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row: Tuple[scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]):
        return (row[0] * row[1])[0, 0]

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data

    def prepare(self):
        df_train = nltk_stemming_without_stopwords(self.input_files['train'])
        df_test = nltk_stemming_without_stopwords(self.input_files['test'])
        train_qs = pd.Series(df_train['question1'].tolist() +
                             df_train['question2'].tolist() +
                             df_test['question1'].tolist() +
                             df_test['question2'].tolist()).astype(str)
        self.vectorizer.fit(train_qs.values)

    def read_data(self, data_file):
        X1 = self.vectorizer.transform(nltk_stemming_without_stopwords(data_file)['question1'].fillna("").tolist())
        X2 = self.vectorizer.transform(nltk_stemming_without_stopwords(data_file)['question2'].fillna("").tolist()).T
        return [(X1.getrow(i), X2.getcol(i)) for i in range(X1.shape[0])]


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
