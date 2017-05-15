import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
        q1_vec = np.zeros((300,))
        q2_vec = np.zeros((300,))
        for word in str(row['question1']).split():
            try:
                q1_vec += self.model[word]
            except:
                continue

        for word in str(row['question2']).split():
            try:
                q2_vec += self.model[word]
            except:
                continue
        return cosine_similarity(q1_vec.reshape(1, -1), q2_vec.reshape(1, -1))


    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('data/input/glove.840B.300d.bin', binary=True)

    def read_data(self, data_file):
        return nltk_tokenize(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()