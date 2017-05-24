import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_stemming
from features.utils import get_stop_words, common_feature_parser


class WordMatchCount(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.count_vectorizer = CountVectorizer(max_df=0.5, min_df=4, ngram_range=(1, 1))
        self.tfidf_transformer = TfidfTransformer(norm=None)
        self.stop_words = get_stop_words()

    @staticmethod
    def get_num_rows(data):
        return data.shape[0]

    def calculate_row_feature(self, row):
        return row

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data

    def prepare(self):
        df_train = nltk_stemming(self.input_files['train'])
        df_test = nltk_stemming(self.input_files['test'])
        train_qs = pd.Series(df_train['question1'].tolist() +
                             df_train['question2'].tolist() +
                             df_test['question1'].tolist() +
                             df_test['question2'].tolist()).astype(str)
        X = self.count_vectorizer.fit_transform(train_qs.values)
        self.tfidf_transformer.fit(X)

    def read_data(self, data_file):
        data = nltk_stemming(data_file)
        q1s = data['question1'].fillna("").tolist()
        q2s = data['question2'].fillna("").tolist()
        X1 = self.count_vectorizer.transform(q1s)
        X2 = self.count_vectorizer.transform(q2s)
        return np.array(self.tfidf_transformer.transform(X1.minimum(X2)).sum(axis=1)).flatten()


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = WordMatchCount(options)
    feature_creator.create()


if __name__ == "__main__":
    main()