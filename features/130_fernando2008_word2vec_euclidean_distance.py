import gensim
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_tokenize
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.vectorizer = TfidfVectorizer()
        self.model = None

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row):
        x1 = row[2]
        x2 = row[3]
        q1_ = [word for word in str(row[0]).split()]
        q2_ = [word for word in str(row[1]).split()]

        q1 = []
        q2 = []
        wq1 = []
        wq2 = []

        for word in q1_:
            try:
                wq1.append(self.model[word])
                q1.append(word)
            except:
                continue
        for word in q2_:
            try:
                wq2.append(self.model[word])
                q2.append(word)
            except:
                continue

        distance = np.zeros((len(wq1), len(wq2)))
        idf1 = np.zeros(len(wq1))
        idf2 = np.zeros(len(wq2))
        for i1, w1 in enumerate(wq1):
            for i2, w2 in enumerate(wq2):
                d = w1 - w2
                distance[i1, i2] = math.sqrt(np.dot(d, d))
        vocab = self.vectorizer.vocabulary_
        for i, w in enumerate(q1):
            if w in vocab:
                idf1[i] = x1[0, vocab[w]]
            else:
                idf1[i] = 0.01
        for i, w in enumerate(q2):
            if w in vocab:
                idf2[i] = x2[0, vocab[w]]
            else:
                idf2[i] = 0.01

        sim1 = 0
        for i1 in range(len(wq1)):
            idf_minimum = 1e10
            for i2 in range(len(wq2)):
                idf_minimum = min(idf_minimum, distance[i1, i2])
            sim1 += idf_minimum * idf1[i1]
        sim1 /= idf1.sum()

        sim2 = 0
        for i2 in range(len(wq2)):
            idf_minimum = 1e10
            for i1 in range(len(wq1)):
                idf_minimum = min(idf_minimum, distance[i1, i2])
            sim2 += idf_minimum * idf2[i2]
        sim2 /= idf2.sum()
        return (sim1 + sim2) / 2

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data

    def prepare(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('data/input/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        df_train = nltk_tokenize(self.input_files['train'])
        df_test = nltk_tokenize(self.input_files['test'])
        train_qs = pd.Series(df_train['question1'].tolist() +
                             df_train['question2'].tolist() +
                             df_test['question1'].tolist() +
                             df_test['question2'].tolist()).astype(str)
        self.vectorizer.fit(train_qs.values)

    def read_data(self, data_file):
        df = nltk_tokenize(data_file)
        X1 = self.vectorizer.transform(df['question1'].fillna("").tolist())
        X2 = self.vectorizer.transform(df['question2'].fillna("").tolist())
        X1rows = [X1.getrow(i) for i in tqdm(range(X1.shape[0]))]
        X2rows = [X2.getrow(i) for i in tqdm(range(X2.shape[0]))]
        return list(zip(df['question1'].tolist(), df['question2'].tolist(), X1rows, X2rows))


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
