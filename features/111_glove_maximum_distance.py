import gensim
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_tokenize
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.model = None

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row_):

        row = row_[1]
        q1 = [word for word in str(row['question1']).split()]
        q2 = [word for word in str(row['question2']).split()]

        wq1 = []
        wq2 = []
        for word in q1:
            try:
                wq1.append(self.model[word])
            except:
                continue
        for word in q2:
            try:
                wq2.append(self.model[word])
            except:
                continue

        maximum = 0
        for q1_, w1 in zip(q1, wq1):
            minimum = 1e10
            for q2_, w2 in zip(q2, wq2):
                minimum = min(minimum, euclidean(w1, w2))
            maximum = max(maximum, minimum)

        for w2 in wq2:
            minimum = 1e10
            for w1 in wq1:
                minimum = min(minimum, euclidean(w1, w2))
            maximum = max(maximum, minimum)
        return maximum

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('data/input/glove.840B.300d.txt', binary=False)

    def read_data(self, data_file):
        return nltk_tokenize(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()


