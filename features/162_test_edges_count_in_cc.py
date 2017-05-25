from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.graph_utils import UnionFind
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.union_find = UnionFind()
        self.edge_total_count = defaultdict(int)
        self.edge_test_count = defaultdict(int)

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, q in tqdm(enumerate(data.question1.astype(str))):
            x = self.union_find.find(q)
            values[i] = self.edge_test_count[x]
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        files = self.input_files
        for k, file in files.items():
            df = pd.read_csv(file)
            for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
                self.union_find.unite(q1, q2)

        train_df = pd.read_csv(files['train'])
        for i, (q1, q2) in tqdm(enumerate(zip(train_df.question1.astype(str), train_df.question2.astype(str)))):
            x = self.union_find.find(q1)
            c = self.edge_total_count[x]
            self.edge_total_count[x] = c + 1
        test_df = pd.read_csv(files['train'])
        for i, (q1, q2) in tqdm(enumerate(zip(test_df.question1.astype(str), test_df.question2.astype(str)))):
            x = self.union_find.find(q1)
            c = self.edge_test_count[x]
            self.edge_test_count[x] = c + 1
            c = self.edge_total_count[x]
            self.edge_total_count[x] = c + 1

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
