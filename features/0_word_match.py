import os

import pandas as pd
import sys

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix, get_stop_words


def word_match_share(row):
    q1words = {}
    q2words = {}
    swords = get_stop_words()

    for word in str(row['question1']).lower().split():
        if word not in swords:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in swords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))


def create_word_match_feature(data_file):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = pd.read_csv(data_file)
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
