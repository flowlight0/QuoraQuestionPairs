import os
import sys

import gensim
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from features.transform import nltk_tokenize
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_word_match_feature(data_file, model: gensim.models.Doc2Vec):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_tokenize(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])

    values = []
    for i, (q1, q2) in tqdm(enumerate(zip(df.question1.values.astype(str).tolist(), df.question2.values.astype(str).tolist()))):
        x1 = model.infer_vector(q1.split(" "))
        x2 = model.infer_vector(q2.split(" "))
        values.append(cosine_similarity(x1.reshape(1, -1), x2.reshape(1, -1))[0, 0])
    df[column_name] = values
    print('Started to write dataset')
    start_time = time.time()
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')
    print('Finished to write dataset: {:.3f} [s]'.format(time.time() - start_time))


def main():
    m = gensim.models.Doc2Vec.load('data/input/enwiki_dbow/doc2vec.bin')
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_word_match_feature(data_file=file_name, model=m)


if __name__ == "__main__":
    main()
