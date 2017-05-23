import os

import joblib
import pandas as pd
from tqdm import tqdm

from features.utils import common_feature_parser, generate_filename_from_prefix


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    dump_graph(options)


def dump_graph(options):
    node_file = get_node_filename(options)
    edge_file = get_edge_filename(options)
    if os.path.exists(node_file) and os.path.exists(edge_file):
        print("File exists {} and {}".format(node_file, edge_file))
        return

    train_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    test_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['test'])
    edges, question2id = build_graph(test_df, train_df)
    dump_load_file(node_file, question2id)
    with open(edge_file, 'w') as f:
        for e in edges:
            print(e, file=f)


def dump_load_file(node_file, question2id):
    joblib.dump(question2id, node_file)


def get_edge_filename(options):
    edge_file = options.data_prefix + 'graph.edges.txt'
    return edge_file


def get_node_filename(options):
    node_file = options.data_prefix + 'graph.nodes.pkl'
    return node_file


def load_node_file(node_filename):
    return joblib.load(node_filename)


def build_graph(test_df, train_df):
    next_id = 0
    question2id = {}
    edges = []
    for i, row in tqdm(train_df.iterrows()):
        qid1 = int(row['qid1'])
        qid2 = int(row['qid2'])
        question2id[row['question1']] = qid1
        question2id[row['question2']] = qid2
        next_id = max(qid1, qid2) + 1
        edges.append("{:d} {:d}".format(qid1, qid2))
    for i, row in tqdm(test_df.iterrows()):
        if row['question1'] not in question2id:
            question2id[row['question1']] = next_id
            next_id += 1
        if row['question2'] not in question2id:
            question2id[row['question2']] = next_id
            next_id += 1
        edges.append("{:d} {:d}".format(question2id[row['question1']], question2id[row['question2']]))
    return edges, question2id


if __name__ == "__main__":
    main()
