import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options, ratio=7):
        super().__init__(options)
        self.question2id = {}
        self.ratio = ratio

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row):
        qid1 = self.question2id[row[1]['question1']]
        qid2 = self.question2id[row[1]['question2']]
        return min(qid1, qid2)

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.question2id = {}
        train_df = pd.read_csv(self.input_files['train'])

        next_id = 0
        for i, row in tqdm(train_df.iterrows()):
            qid1 = float(row['qid1'])
            qid2 = float(row['qid2'])
            self.question2id[row['question1']] = qid1
            self.question2id[row['question2']] = qid2
            next_id = max(qid1, qid2) + 1

        test_df = pd.read_csv(self.input_files['test'])

        for i, row in tqdm(test_df.iterrows()):
            if row['question1'] not in self.question2id:
                self.question2id[row['question1']] = next_id
                next_id += 1.0 / self.ratio
            if row['question2'] not in self.question2id:
                self.question2id[row['question2']] = next_id
                next_id += 1.0 / self.ratio

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
