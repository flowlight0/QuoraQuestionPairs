import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.neighbor_sets = {}
        self.value_cache = {}

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_row_feature(self, row):
        q1 = str(row[1]['question1'])

        if q1 in self.value_cache:
            return self.value_cache[q1]

        size = len(self.neighbor_sets[q1])
        if size < 2:
            return 0
        else:
            count = 0
            for q2, q3 in itertools.combinations(self.neighbor_sets[q1], 2):
                if q2 in self.neighbor_sets[q3]:
                    count += 1
            answer = 2.0 * count / ((size - 1) * size)
            self.value_cache[q1] = answer
            return answer

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        self.neighbor_sets = {}
        for k, file in self.input_files.items():
            print(file)
            self.preprocess(pd.read_csv(file))

    def preprocess(self, df: pd.DataFrame):
        for i, row in tqdm(df.iterrows()):
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            if q1 not in self.neighbor_sets:
                self.neighbor_sets[q1] = set()
            if q2 not in self.neighbor_sets:
                self.neighbor_sets[q2] = set()
            self.neighbor_sets[q1].add(q2)
            self.neighbor_sets[q2].add(q1)

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
