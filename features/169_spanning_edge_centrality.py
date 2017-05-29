import os
import subprocess

import pandas as pd

from features.feature_template import RowWiseFeatureCreatorBase
from features.graph_dumper import dump_graph, get_node_filename, get_edge_filename
from features.utils import common_feature_parser


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        dump_graph(options)
        self.data = {}
        self.node_filename = get_node_filename(options)
        self.edge_filename = get_edge_filename(options)
        self.column_name = "f" + os.path.basename(__file__).split("_")[0]
        self.num_samples = 5000

    @staticmethod
    def get_num_rows(data):
        return len(data)

    def calculate_features(self, data):
        return data[self.column_name].values

    @staticmethod
    def get_row_wise_iterator(data):
        return data.iterrows()

    def prepare(self):
        binary_file = os.path.join(os.path.dirname(__file__), '../graph/spanning_edge_centrality_feature_main')
        program_file = binary_file + '.cpp'
        temp_file = __file__ + '.feature.tmp'
        commands = ['g++', '-O3', '-std=c++11', program_file, '-o', binary_file]
        subprocess.call(commands)
        commands = [binary_file, self.edge_filename, temp_file, self.column_name, str(self.num_samples)]
        subprocess.call(commands)

        train_df = pd.read_csv(self.input_files['train'])
        temp_df = pd.read_csv(temp_file)
        self.data[self.input_files['train']] = temp_df.iloc[:train_df.shape[0]]
        self.data[self.input_files['test']] = temp_df.iloc[train_df.shape[0]:]

    def read_data(self, data_file):
        return self.data[data_file]


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
