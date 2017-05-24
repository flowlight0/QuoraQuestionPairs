import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.core_number = {}

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_features(self, data):
        values = np.zeros(self.get_num_rows(data))
        for i, q in tqdm(enumerate(data.question2.astype(str))):
            values[i] = self.core_number[q]
        return values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        g = nx.Graph()
        for k, file in self.input_files.items():
            df = pd.read_csv(file)
            a = pd.DataFrame()
            a.iterrows()
            for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
                g.add_edge(q1, q2)
        g.remove_edges_from(g.selfloop_edges())
        self.core_number = nx.core_number(g)

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
