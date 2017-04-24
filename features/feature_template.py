import os
import sys
import time
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.utils import feature_output_file, generate_filename_from_prefix


class RowWiseFeatureCreatorBase:
    def __init__(self, options):
        self.input_files = dict(generate_filename_from_prefix(options.data_prefix))
        self.train_only = options.train_only
        self.n_threads = options.n_threads

    def prepare(self):
        raise NotImplementedError

    def read_data(self, data_file):
        raise NotImplementedError

    @staticmethod
    def get_row_wise_iterator(data):
        raise NotImplementedError

    @staticmethod
    def get_num_rows(data):
        raise NotImplementedError

    def calculate_row_feature(self, row):
        raise NotImplementedError

    def calculate_features(self, data):
        pool = Pool(self.n_threads)
        values = np.zeros(self.get_num_rows(data))
        for i, value in tqdm(enumerate(pool.map(self.calculate_row_feature, self.get_row_wise_iterator(data)))):
            values[i] = value
        return values

    def create_features(self, input_file):
        output_file = feature_output_file(input_file)

        start_time = time.time()
        print("Start to create features {} {}".format(sys.argv[0], input_file), file=sys.stderr)
        if os.path.exists(output_file):
            print('File exists {}.'.format(feature_output_file(output_file)))
            return

        data = self.read_data(input_file)
        values = self.calculate_features(data)
        print("Finished to create features {} {}: {:.2f} [s]".format(sys.argv[0], input_file, time.time() - start_time),
              file=sys.stderr)

        start_time = time.time()
        print("Start to write features {} {}".format(sys.argv[0], input_file), file=sys.stderr)
        column_name = 'f{0}'.format(os.path.basename(feature_output_file(input_file)).split('_')[0])
        df = pd.DataFrame()
        df[column_name] = values
        df[[column_name]].to_csv(output_file, index=False, float_format='%.5f')
        print("Finished to write features {} {}: {:.2f} [s]".format(sys.argv[0], input_file, time.time() - start_time),
              file=sys.stderr)

    def create(self):
        start_time = time.time()
        print("Start to pre-computation for creating feature {}".format(sys.argv[0]), file=sys.stderr)
        self.prepare()
        print("Finished to pre-computation for creating feature {}: {:.2f} [s]"
              .format(sys.argv[0], time.time() - start_time), file=sys.stderr)
        self.create_features(self.input_files['train'])
        if not self.train_only:
            self.create_features(self.input_files['test'])
