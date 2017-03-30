import argparse
import os
import sys

from nltk.corpus import stopwords


def get_stop_words():
    return set(stopwords.words("english"))


def feature_output_file(data_file, feature_creator_file=None):
    if feature_creator_file is None:
        return _feature_output_file(data_file, sys.argv[0])
    else:
        return _feature_output_file(data_file, feature_creator_file)


def _get_feature_file(data_file, feature_file_id):
    (head, basename) = os.path.split(data_file)
    (prefix, datadir) = os.path.split(head)
    return os.path.join(prefix, 'working', str(feature_file_id) + '_' + basename)


def _check_feature_file(data_file: str):
    if not os.path.exists(data_file):
        print("Data file {0} doesn't exist.".format(data_file))
        sys.exit(-1)


def _feature_output_file(data_file: str, feature_creator_file: str):
    _check_feature_file(data_file)

    feature_creator_file_id = os.path.basename(feature_creator_file).split("_")[0]
    if not feature_creator_file_id.isdigit():
        print("Feature creator file {} must starts with '(feature_id)_'."
              .format(feature_creator_file).format(file=sys.stderr))
        sys.exit(-1)
    return _get_feature_file(data_file, feature_creator_file_id)


def feature_input_file(data_file, feature_file_id):
    _check_feature_file(data_file)
    return _get_feature_file(data_file, feature_file_id)


def generate_filename_from_prefix(prefix: str):
    for k, suffix in [('train', 'train.csv'), ('test', 'test.csv')]:
        yield (k, prefix + suffix)


def common_feature_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', default='../data/input/', type=str)
    return parser
