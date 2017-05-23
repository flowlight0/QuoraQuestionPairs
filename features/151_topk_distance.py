import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_template import RowWiseFeatureCreatorBase
from features.graph_dumper import get_node_filename, get_edge_filename
from features.utils import common_feature_parser, feature_output_file


class FeatureCreator(RowWiseFeatureCreatorBase):
    def __init__(self, options):
        super().__init__(options)
        self.node_filename = get_node_filename(options)
        self.edge_filename = get_edge_filename(options)
        self.topk_filename = options.data_prefix + 'graph.top_32'
        if not os.path.exists(self.node_filename) or not os.path.exists(self.edge_filename) or not os.path.exists(
                self.topk_filename):
            raise FileNotFoundError("You should create graph before using this script. "
                                    "Please ask takanori about how to generate graph files")
        self.features = {}

    def create(self):
        train_output_file = feature_output_file(self.input_files['train'])
        test_output_file = feature_output_file(self.input_files['test'])
        if os.path.exists(train_output_file) and os.path.exists(test_output_file):
            print('File exists {} and {}.'.format(train_output_file, test_output_file))
            return

        topk_df = pd.read_csv(self.topk_filename)
        train_df = pd.read_csv(self.input_files['train'])
        self.features['train'] = topk_df.iloc[:train_df.shape[0]]
        self.features['test'] = topk_df.iloc[train_df.shape[0]:]
        self.features['train'].to_csv(train_output_file, index=False)
        self.features['test'].to_csv(test_output_file, index=False)

    def read_data(self, data_file):
        return pd.read_csv(data_file)


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = FeatureCreator(options)
    feature_creator.create()


if __name__ == "__main__":
    main()
