import numpy as np
from gensim.models.wrappers import FastText
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
        return self.model.wmdistance(q1, q2)

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.model = FastText.load_fasttext_format('data/input/wiki.en')
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
