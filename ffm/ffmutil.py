import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


FFM_FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'ffm_features')
LIBFFM_DIR = os.path.join(os.path.dirname(__file__), 'libffm')
FFM_WORK_DIR = os.path.join(os.path.dirname(__file__), 'work')


def __create_filepath(train_or_test, field_name):
    assert(train_or_test in ('train', 'test'))

    return os.path.join(FFM_FEATURES_DIR, '{}_{}'.format(train_or_test, field_name))


def create_ffm_field_file(data, train_or_test, field_name):
    assert(train_or_test in ('train', 'test'))

    with open(__create_filepath(train_or_test, field_name), 'wb') as f:
        pickle.dump(data, file=f)


def read_ffm_field_file(train_or_test, field_name):
    with open(__create_filepath(train_or_test, field_name), 'rb') as f:
        return pickle.load(file=f)


def __create_one_field_str(field_index, field_categories):
    return ' '.join('{}:{}:1'.format(field_index, cate) for cate in field_categories)


def create_ffm_file(train_or_test, fields, ffm_filename):
    assert (train_or_test in ('train', 'test'))

    input_data = pd.read_csv('../data/input/{}.csv'.format(train_or_test))
    data_size = len(input_data)

    num_fields = len(fields)

    print('reading ffm fields')
    ffm_fields = [[] for _ in range(data_size)]
    for field in tqdm(fields):
        ffm_field = read_ffm_field_file(train_or_test, field)
        for i in range(data_size):
            ffm_fields[i].append(ffm_field[i])

    print('generating ffm file')
    lines = []
    for ffm_fields_in_row in tqdm(ffm_fields):
        assert(len(ffm_fields_in_row) == num_fields)

        line = ' '.join(
            __create_one_field_str(field_index, field_categories)
            for field_index, field_categories in zip(range(num_fields), ffm_fields_in_row)
        )
        lines.append(line)

    if train_or_test == 'train':
        labels = input_data.is_duplicate.tolist()
    else:
        labels = [0] * data_size

    lines = ['{} {}'.format(label, line).rstrip() for label, line in zip(labels, lines)]
    filepath = os.path.join(FFM_WORK_DIR, '{}_{}.ffm'.format(train_or_test, ffm_filename))
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    return filepath


def train(model_name, train_ffm_file, validation_ffm_file):
    model_filepath = os.path.join(FFM_WORK_DIR, model_name + '.model')
    subprocess.call([
        os.path.join(LIBFFM_DIR, 'ffm-train'),
        '-s', '12',
        '--auto-stop',
        '-p', validation_ffm_file,
        train_ffm_file,
        model_filepath
    ])
    return model_filepath


def predict(model_filepath, test_ffm_file):
    output_filepath = os.path.join(FFM_WORK_DIR, 'temp.pred')
    subprocess.call([
        os.path.join(LIBFFM_DIR, 'ffm-predict'),
        test_ffm_file,
        model_filepath,
        output_filepath
    ])
    return pd.read_csv(output_filepath, header=None, names=['pred']).pred.tolist()


class FFMCV:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cv_infos = None
        self.model_filepaths = None

    def __generate_cv_data(self, fields):
        ffm_filepath = create_ffm_file('train', fields, ffm_filename=self.model_name)

        with open(ffm_filepath) as f:
            lines = f.readlines()
        train_df = pd.read_csv('../data/input/train.csv')
        assert(len(lines) == len(train_df))

        def write_sub_lines(filepath, index):
            sub_lines = [lines[i] for i in index]
            with open(filepath, 'w') as f:
                f.writelines(sub_lines)

        y = train_df.is_duplicate
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=114514)
        self.cv_infos = []
        for i, (train_index, valid_index) in enumerate(skf.split(np.zeros(len(y)), y)):
            train_filepath = '{}.train{}'.format(ffm_filepath, i)
            valid_filepath = '{}.valid{}'.format(ffm_filepath, i)

            write_sub_lines(train_filepath, train_index)
            write_sub_lines(valid_filepath, valid_index)

            self.cv_infos.append({
                'train_filepath': train_filepath,
                'valid_filepath': valid_filepath,
                'train_index': train_index,
                'valid_index': valid_index
            })

    def __do_train_cv(self):
        train_df = pd.read_csv('../data/input/train.csv')

        self.model_filepaths = []
        preds = [-1] * len(train_df)
        for cv_i, cv_infos in enumerate(self.cv_infos):
            model_filepath = train(self.model_name + str(cv_i), cv_infos['train_filepath'], cv_infos['valid_filepath'])
            self.model_filepaths.append(model_filepath)

            pre = predict(model_filepath, cv_infos['valid_filepath'])
            for i, p in zip(cv_infos['valid_index'], pre):
                preds[i] = p
        return preds

    def train(self, fields):
        self.__generate_cv_data(fields)
        preds = self.__do_train_cv()
        return preds

    def predict(self, test_ffm_file):
        preds = []
        for model_filepath in self.model_filepaths:
            preds.append(predict(model_filepath, test_ffm_file))
        mean_p = np.array(preds).transpose().mean(axis=1)
        return mean_p