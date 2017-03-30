import os
import sys

import gensim

from features.transform import nltk_tokenize
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix, get_stop_words


def wmd(row, model: gensim.models.KeyedVectors):
    swords = get_stop_words()
    q1 = [word for word in str(row['question1']).split() if word not in swords]
    q2 = [word for word in str(row['question2']).split() if word not in swords]
    return model.wmdistance(q1, q2)


def create_word_match_feature(data_file, model: gensim.models.KeyedVectors):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = nltk_tokenize(data_file)

    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(wmd, axis=1, raw=True, model=model)
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    model = gensim.models.KeyedVectors.load_word2vec_format('data/input/GoogleNews-vectors-negative300.bin', binary=True)
    model.init_sims(replace=True)
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_word_match_feature(data_file=file_name, model=model)


if __name__ == "__main__":
    main()
