import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_tokenize
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.vocab = None
        self.model = None

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row_):
        row = row_[1]
        q = str(row['question2']).split(' ')
        return sum([word.isalnum() and word not in self.vocab for word in q])

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        df_train = nltk_tokenize('data/input/train.csv')
        df_test = nltk_tokenize('data/input/test.csv')
        vectorizer = CountVectorizer(min_df=2)
        qs = pd.Series(df_train.question1.astype(str).tolist() + df_train.question2.astype(str).tolist() +
                       df_test.question1.astype(str).tolist() + df_test.question2.astype(str).tolist())
        vectorizer.fit(qs)
        self.vocab = vectorizer.vocabulary_

    def read_data(self, data_file):
        return nltk_tokenize(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
