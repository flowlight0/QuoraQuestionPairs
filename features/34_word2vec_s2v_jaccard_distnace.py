import os
import sys

import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard

from features.transform import sentence2vec
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_feature(data_file, model: gensim.models.KeyedVectors):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    question1_vectors, question2_vectors = sentence2vec(data_file)

    print(sys.argv[0], data_file, file=sys.stderr)
    df = pd.DataFrame()
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    values = []
    tdqm
    df[column_name] = np.nan_to_num([jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                     np.nan_to_num(question2_vectors))])

    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('data/input/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, model=model)


if __name__ == "__main__":
    main()
