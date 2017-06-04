import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def calculate_q1_weight_avg(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        values[i] = np.mean([neighbor_weights[q1][q] for q in neighbor_sets[q1]])
    return values


def calculate_q2_weight_avg(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        values[i] = np.mean([neighbor_weights[q2][q] for q in neighbor_sets[q2]])
    return values


def calculate_q1_q2_intersection_weight_sum(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        values[i] = np.sum([neighbor_weights[q1][q] * neighbor_weights[q2][q]
                            for q in q1_neighbors.intersection(q2_neighbors)])
    return values


def calculate_q1_q2_intersection_weight_avg(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        intersect = q1_neighbors.intersection(q2_neighbors)
        if len(intersect) > 0:
            values[i] = np.mean([neighbor_weights[q1][q] * neighbor_weights[q2][q] for q in intersect])
        else:
            values[i] = -1
    return values


def calculate_q1_q2_intersection_weight_max(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        intersect = q1_neighbors.intersection(q2_neighbors)
        if len(intersect) > 0:
            values[i] = np.max([neighbor_weights[q1][q] * neighbor_weights[q2][q] for q in intersect])
        else:
            values[i] = -1
    return values


def calculate_q1_q2_intersection_weight_min(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        intersect = q1_neighbors.intersection(q2_neighbors)
        if len(intersect) > 0:
            values[i] = np.min([neighbor_weights[q1][q] * neighbor_weights[q2][q] for q in intersect])
        else:
            values[i] = -1
    return values


def calculate_q1_q2_intersection_weight_diff_max(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        intersect = q1_neighbors.intersection(q2_neighbors)
        if len(intersect) > 0:
            values[i] = np.max([max(neighbor_weights[q1][q] * (1 - neighbor_weights[q2][q]),
                                    (1 - neighbor_weights[q1][q]) * neighbor_weights[q2][q]) for q in intersect])
        else:
            values[i] = -1
    return values


def calculate_q1_q2_intersection_weight_diff_min(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        q1_neighbors = neighbor_sets[q1]
        q2_neighbors = neighbor_sets[q2]
        intersect = q1_neighbors.intersection(q2_neighbors)
        if len(intersect) > 0:
            values[i] = np.min([max(neighbor_weights[q1][q] * (1 - neighbor_weights[q2][q]),
                                    (1 - neighbor_weights[q1][q]) * neighbor_weights[q2][q]) for q in intersect])
        else:
            values[i] = -1
    return values


def calculate_identity(df, neighbor_sets, neighbor_weights):
    values = np.zeros(df.shape[0])
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str)))):
        assert (q2 in neighbor_sets[q1])
        assert (q1 in neighbor_sets[q2])
        values[i] = neighbor_weights[q1][q2]
    return values


def create_feature(data_file, neighbor_sets: defaultdict, neighbor_weights: defaultdict):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = pd.read_csv(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)

    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    column_names = []
    features = [
        ('identity', calculate_identity),
        ('q1_weight_avg', calculate_q1_weight_avg),
        ('q2_weight_avg', calculate_q2_weight_avg),
        ('q1_q2_intersection_weight_sum', calculate_q1_q2_intersection_weight_sum),
        ('q1_q2_intersection_weight_avg', calculate_q1_q2_intersection_weight_avg),
        ('q1_q2_intersection_weight_max', calculate_q1_q2_intersection_weight_max),
        ('q1_q2_intersection_weight_min', calculate_q1_q2_intersection_weight_min),
        ('q1_q2_intersection_weight_diff_max', calculate_q1_q2_intersection_weight_diff_max),
        ('q1_q2_intersection_weight_diff_min', calculate_q1_q2_intersection_weight_diff_min)
    ]

    for (column_name_suffix, feature_calculator) in features:
        column_name = column_name_prefix + '.' + column_name_suffix
        df[column_name] = feature_calculator(df, neighbor_sets, neighbor_weights)
        column_names.append(column_name)
    column_names = pd.Index(column_names)
    df[column_names].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def prepare_graph(options, file_prefix):
    input_files = dict(generate_filename_from_prefix(options.data_prefix))
    train_file = os.path.join(os.path.dirname(input_files['train']), '../output/', file_prefix + '.model.train.pred')
    test_file = os.path.join(os.path.dirname(input_files['train']), '../output/', file_prefix + '.submission.csv')
    print('Stacking ingredients: {} and {}'.format(train_file, test_file), file=sys.stderr)
    neighbor_sets = defaultdict(set)
    neighbor_weights = defaultdict(dict)

    dfs = []
    df_train = pd.read_csv(input_files['train'])
    df_train['prob'] = pd.read_csv(train_file)['prediction']
    dfs.append(df_train)

    df_test = pd.read_csv(input_files['test'])
    df_test['prob'] = pd.read_csv(test_file)['is_duplicate']
    dfs.append(df_test)

    for df in dfs:
        for i, (q1, q2, value) in tqdm(enumerate(zip(df.question1.astype(str), df.question2.astype(str), df.prob))):
            neighbor_sets[q1].add(q2)
            neighbor_weights[q1][q2] = value
            neighbor_sets[q2].add(q1)
            neighbor_weights[q2][q1] = value
    return neighbor_sets, neighbor_weights


def main():
    file_prefix = 'gbm_cross_1013.json.gbm_cross_1013.json'
    options = common_feature_parser().parse_args()
    neighbor_sets, neighbor_weights = prepare_graph(options, file_prefix)
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, neighbor_sets=neighbor_sets, neighbor_weights=neighbor_weights)


if __name__ == "__main__":
    main()
