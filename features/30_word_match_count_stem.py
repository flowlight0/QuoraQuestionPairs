import os
import sys

from features.transform import nltk_stemming
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix, get_stop_words


def word_match_share(row):
    swords = get_stop_words()
    q1set = set([word for word in str(row['question1']).split() if word not in swords])
    q2set = set([word for word in str(row['question2']).split() if word not in swords])
    return len(q1set.intersection(q2set))


def create_word_match_feature(data_file):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = nltk_stemming(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(word_match_share, axis=1, raw=True)
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_word_match_feature(data_file=file_name)


if __name__ == "__main__":
    main()
