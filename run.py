import argparse
import glob
import json
import os
import shlex
import subprocess

import pandas as pd
import sys

from features.utils import feature_input_file, generate_filename_from_prefix, feature_output_file


def get_tmp_train_file():
    return 'tmp_train.txt'


def get_tmp_scaled_train_file():
    return 'tmp_train.scaled.txt'


def get_tmp_test_file():
    return 'tmp_test.txt'


def get_feature_creator_file(feature_id: int):
    files = glob.glob('./features/*.py', recursive=False)
    for file in files:
        if os.path.basename(file).startswith(str(feature_id) + '_'):
            return file
    raise FileNotFoundError("File that matches feature_id ({}) doesn't exist".format(feature_id))


def join_dataset(data_file: str, features_ids: list, tmp_file: str, is_train: bool):
    feature_files = [feature_input_file(data_file, feature_id) for feature_id in features_ids]
    with open(tmp_file, 'w') as f:
        subprocess.call(shlex.split('paste -d, {0}'.format(" ".join(feature_files))), stdout=f)

    if is_train:
        data = pd.read_csv(tmp_file)
        data['y'] = pd.read_csv(data_file)['is_duplicate'].values
        return data
    else:
        return pd.read_csv(tmp_file)


def scale_dataset(data, target_positive_ratio=0.165):
    if target_positive_ratio > 0:
        pos_data = data[data.y == 1]
        neg_data = data[data.y == 0]
        size = int(len(pos_data) / target_positive_ratio) - len(pos_data)
        while len(neg_data) < size:
            neg_data = pd.concat([neg_data, neg_data])
        neg_data = neg_data[:size]
        scaled_data = pd.concat([pos_data, neg_data], ignore_index=True)
    else:
        scaled_data = data
    scaled_data.to_csv(get_tmp_scaled_train_file(), index=False)


def check_feature_existence(feature_creator_file, data_prefix):
    for k, file_name in generate_filename_from_prefix(data_prefix):
        if not os.path.exists(feature_output_file(file_name, feature_creator_file)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config/xgb_0.json')
    options = parser.parse_args()
    config = json.load(open(options.config_file))

    for feature_id in config['features']:
        feature_creator_file = get_feature_creator_file(feature_id)
        if check_feature_existence(feature_creator_file, config['data_prefix']):
            print('Feature file for {} exists.'.format(feature_creator_file), file=sys.stderr)
            continue
        subprocess.call(["python3", feature_creator_file, "--data_prefix", config['data_prefix']])

    data_files = dict(generate_filename_from_prefix(config['data_prefix']))
    scale_dataset(join_dataset(data_files['train'], config['features'], get_tmp_train_file(), True))

    model_python = config['model']['path']

    prefix = os.path.basename(options.config_file) + '.{}'.format(os.path.basename(options.config_file))
    model_file = os.path.join('data/output', prefix + '.model')
    output_file = os.path.join('data/output', prefix + '.stats.json')

    subprocess.call(['python3', model_python, '--data_file', get_tmp_scaled_train_file(),
                     '--config_file', options.config_file, '--model_file', model_file, '--train',
                     '--log_file', output_file])

    join_dataset(data_files['test'], config['features'], get_tmp_test_file(), False)
    submission_file = os.path.join('data/output', prefix + '.submission.csv')
    subprocess.call(['python3', model_python, '--data_file', get_tmp_test_file(),
                     '--config_file', options.config_file, '--model_file', model_file,
                     '--submission_file', submission_file])


if __name__ == "__main__":
    main()
