import os
from collections import Counter

import numpy as np
import pandas as pd

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix, get_stop_words


def get_weights(train_qs):
    def get_weight(count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    eps = 5000
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    return {word: get_weight(count, eps=eps) for word, count in counts.items()}


def tfidf_word_match_share(row, weights):
    swords = get_stop_words()
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in swords:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in swords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + \
                     [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    den = np.sum(total_weights)
    return np.sum(shared_weights) / den if den > 0 else 0


def create_word_match_feature(data_file, weights):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = pd.read_csv(data_file)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(tfidf_word_match_share, axis=1, raw=True, weights=weights)
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    df_train = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    df_test = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['test'])
    train_qs = pd.Series(df_train['question1'].tolist() +
                         df_train['question2'].tolist() +
                         df_test['question1'].tolist() +
                         df_test['question2'].tolist()).astype(str)
    weights = get_weights(train_qs)
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_word_match_feature(data_file=file_name, weights=weights)


if __name__ == "__main__":
    main()
