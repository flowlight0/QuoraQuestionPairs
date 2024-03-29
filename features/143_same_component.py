import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parent[x] = y


def create_feature(data_file, features: pd.DataFrame):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    features.to_csv(feature_output_file(data_file), index=False)


def calculate_features(df, uf: UnionFind):
    values = []
    feature_id = 'f' + os.path.basename(str(__file__)).split("_")[0] + '.cat'
    for i, row in tqdm(df.iterrows()):
        if uf.same(str(row['question1']), str(row['question2'])):
            values.append(1)
        else:
            uf.unite(str(row['question1']), str(row['question2']))
            values.append(0)
    return pd.DataFrame(data=values, columns=[feature_id])


def main():
    options = common_feature_parser().parse_args()
    # from https://www.kaggle.com/jturkewitz/magic-features-0-03-gain/notebook
    train_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    test_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['test'])
    uf = UnionFind()
    features = {'train': calculate_features(train_df, uf), 'test': calculate_features(test_df, uf)}

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, features=features[k])


if __name__ == "__main__":
    main()
