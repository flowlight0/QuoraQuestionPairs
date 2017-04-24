import os
import sys

import gensim
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from features.transform import nltk_tokenize
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix, get_stop_words


def calculate_distance(q1: str, q2: str, model: gensim.models.KeyedVectors):
    swords = get_stop_words()
    q1 = [word for word in str(q1).lower().split() if word not in swords and word.isalpha()]
    q2 = [word for word in str(q2).lower().split() if word not in swords and word.isalpha()]

    wq1 = []
    wq2 = []
    for word in q1:
        try:
            wq1.append(model[word])
        except:
            continue
    for word in q2:
        try:
            wq2.append(model[word])
        except:
            continue

    maximum = 0
    for w2 in wq2:
        minimum = 1e10
        for w1 in wq1:
            minimum = min(minimum, euclidean(w1, w2))
        maximum = max(maximum, minimum)
    return maximum


def create_feature(data_file, model):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_tokenize(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])

    values = np.zeros((df.shape[0]))
    for i in tqdm(range(df.shape[0])):
        q1 = df.question1.values[i]
        q2 = df.question2.values[i]
        values[i] = calculate_distance(q1, q2, model)

    df[column_name] = values
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    parser = common_feature_parser()
    parser.add_argument('--google_word2vec', default='data/input/GoogleNews-vectors-negative300.bin', type=str)
    options = parser.parse_args()

    model = gensim.models.KeyedVectors.load_word2vec_format(options.google_word2vec, binary=True)
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        if 'test' in file_name and options.train_only:
            continue
        create_feature(data_file=file_name, model=model)


if __name__ == "__main__":
    main()
