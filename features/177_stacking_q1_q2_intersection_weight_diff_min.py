import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.neighbor_sets = defaultdict(set)
        self.neighbor_weights = defaultdict(dict)

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_features(self, df):
        values = np.zeros(self.get_num_rows(df))
        for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
            q1_neighbors = self.neighbor_sets[q1]
            q2_neighbors = self.neighbor_sets[q2]
            intersect = q1_neighbors.intersection(q2_neighbors)
            if len(intersect) > 0:
                values[i] = np.min([max(self.neighbor_weights[q1][q] * (1 - self.neighbor_weights[q2][q]),
                                        (1 - self.neighbor_weights[q1][q]) * self.neighbor_weights[q2][q]) for q in intersect])
            else:
                values[i] = -1
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.neighbor_sets = defaultdict(set)
        self.neighbor_weights = defaultdict(dict)
        train_prob_file = os.path.join(os.path.dirname(self.input_files['train']),
                                       '../output/gbm_cross_52.json.gbm_cross_52.json.model.train.pred')
        test_prob_file = os.path.join(os.path.dirname(self.input_files['train']),
                                      '../output/gbm_cross_52.json.gbm_cross_52.json.submission.csv')
        dfs = []
        df_train = pd.read_csv(self.input_files['train'])
        df_train['prob'] = pd.read_csv(train_prob_file)['prediction']
        dfs.append(df_train)

        df_test = pd.read_csv(self.input_files['test'])
        df_test['prob'] = pd.read_csv(test_prob_file)['is_duplicate']
        dfs.append(df_test)

        for df in dfs:
            for i, (q1, q2, value) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str), df.prob))):
                self.neighbor_sets[q1].add(q2)
                self.neighbor_weights[q1][q2] = value
                self.neighbor_sets[q2].add(q1)
                self.neighbor_weights[q2][q1] = value

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
