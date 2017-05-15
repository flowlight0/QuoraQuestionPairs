import argparse
import glob
import joblib
import json
import os
import shlex
import subprocess
import sys

import pandas as pd
import time

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
    scaled_data.to_csv(get_tmp_scaled_train_file(), index=False, float_format='%.5f')
    return scaled_data


def check_feature_existence(feature_creator_file, data_prefix):
    for k, file_name in generate_filename_from_prefix(data_prefix):
        if not os.path.exists(feature_output_file(file_name, feature_creator_file)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./config/xgb_0.json')
    parser.add_argument('--train_only', action='store_true')
    options = parser.parse_args()
    config = json.load(open(options.config_file))

    data_files = generate_data_files(config, options)
    takapt_features = config['takapt_features'] if 'takapt_features' in config else []
    print("Started generating training dataset from features {} and {}".format(config['features'], takapt_features))
    start_time = time.time()
    join_dataset(data_files['train'], config['features'], get_tmp_train_file(), True, config)
    print("Finished generating training dataset: {:.3f}s".format(time.time() - start_time))
    model_python = config['model']['path']
    prefix = os.path.basename(options.config_file) + '.{}'.format(os.path.basename(options.config_file))
    model_file = os.path.join('data/output', prefix + '.model')
    output_file = os.path.join('data/output', prefix + '.stats.json')

    subprocess.call(['python3', model_python, '--data_file', get_tmp_train_file(),
                     '--config_file', options.config_file, '--model_file', model_file, '--train',
                     '--log_file', output_file])

    if not options.train_only:
        print("Started generating testing dataset from features {} and {}".format(config['features'], takapt_features))
        start_time = time.time()
        join_dataset(data_files['test'], config['features'], get_tmp_test_file(), False, config)
        print("Finished generating testing dataset: {:.3f}s".format(time.time() - start_time))

        submission_file = os.path.join('data/output', prefix + '.submission.csv')
        subprocess.call(['python3', model_python, '--data_file', get_tmp_test_file(),
                         '--config_file', options.config_file, '--model_file', model_file,
                         '--submission_file', submission_file])


def generate_data_files(config, options):
    for feature_id in config['features']:
        feature_creator_file = get_feature_creator_file(feature_id)
        if check_feature_existence(feature_creator_file, config['data_prefix']):
            print('Feature file for {} exists.'.format(feature_creator_file), file=sys.stderr)
            continue
        commands = ["python3", feature_creator_file, "--data_prefix", config['data_prefix']]
        if options.train_only:
            commands.append("--train_only")
        subprocess.call(commands)
    data_files = dict(generate_filename_from_prefix(config['data_prefix']))
    return data_files


def join_dataset(data_file: str, features_ids: list, tmp_file: str, is_train: bool, config: dict):
    feature_files = [feature_input_file(data_file, feature_id) for feature_id in features_ids]
    if 'takapt_features' in config:
        df = pd.DataFrame()
        takapt_file = 'tmp_takapt.csv'
        for feature_name in config['takapt_features']:
            data_prefix = config['data_prefix']
            feature_file = os.path.join("data/takapt/feature/", os.path.basename(data_prefix), ("train_" if is_train else "test_") + feature_name + ".pkl")
            feature = joblib.load(feature_file)
            if is_train:
                feature.drop(labels='id', axis=1, inplace=True)
            else:
                feature.drop(labels='test_id', axis=1, inplace=True)
            for column in feature.columns:
                df[column] = feature[column].values
        df.to_csv(takapt_file, index=False)
        feature_files.append(takapt_file)

    with open(tmp_file, 'w') as f:
        subprocess.call(shlex.split('paste -d, {0}'.format(" ".join(feature_files))), stdout=f)

    if is_train:
        data = pd.read_csv(tmp_file)
        data['y'] = pd.read_csv(data_file)['is_duplicate'].values
        data.to_csv(get_tmp_train_file(), index=False, float_format='%.5f')
        print(data.columns)
        return data
    else:
        return pd.read_csv(tmp_file)


if __name__ == "__main__":
    main()
