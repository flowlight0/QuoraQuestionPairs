import gensim
import numpy as np
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_tokenize
from features.utils import common_feature_parser, get_stop_words


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.model = None

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row_):
        row = row_[1]
        swords = get_stop_words()
        q1 = [word for word in str(row['question1']).split() if word not in swords]
        q2 = [word for word in str(row['question2']).split() if word not in swords]

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

        distance = np.zeros((len(wq1), len(wq2)))
        for i1, w1 in enumerate(wq1):
            for i2, w2 in enumerate(wq2):
                distance[i1, i2] = np.dot(w1, w2)

        maximum = 0
        for i1 in range(len(wq1)):
            minimum = 1e10
            for i2 in range(len(wq2)):
                minimum = min(minimum, distance[i1, i2])
            maximum = max(maximum, minimum)

        for i2 in range(len(wq2)):
            minimum = 1e10
            for i1 in range(len(wq1)):
                minimum = min(minimum, distance[i1, i2])
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
        self.model = gensim.models.KeyedVectors.load_word2vec_format('data/input/wiki.en', binary=False)
        self.model.init_sims(replace=True)

    def read_data(self, data_file):
        return nltk_tokenize(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
