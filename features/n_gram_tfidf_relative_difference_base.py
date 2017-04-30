import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_stemming_without_stopwords
from features.utils import get_stop_words


class NGramTfidfRelativeDifference(RowWiseFeatureCreatorBase):
    def __init__(self, options, ngram_range=(1, 1)):
        super().__init__(options)
        self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=8, ngram_range=ngram_range)
        self.stop_words = get_stop_words()

    @staticmethod
    def get_num_rows(data):
        return data.shape[0]

    def calculate_row_feature(self, row):
        if max(abs(row[0]), abs(row[1])) > 0:
            return abs(row[0] - row[1]) / max(abs(row[0]), abs(row[1]))
        else:
            return 0

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
        X1 = np.array(
            self.vectorizer.transform(nltk_stemming_without_stopwords(data_file)['question1'].fillna("").tolist()).sum(
                axis=1)).reshape(-1, 1)
        X2 = np.array(
            self.vectorizer.transform(nltk_stemming_without_stopwords(data_file)['question2'].fillna("").tolist()).sum(
                axis=1)).reshape(-1, 1)
        return np.hstack((X1, X2))

