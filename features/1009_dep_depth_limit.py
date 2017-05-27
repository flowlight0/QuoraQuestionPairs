import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
import en_core_web_md
import os

from features.feature_template import RowWiseFeatureCreatorBase
from features.transform import nltk_tokenize
from features.utils import common_feature_parser
import os
from typing import Tuple

import en_core_web_md
import numpy as np
import pandas as pd
import spacy
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from features.utils import feature_output_file, common_feature_parser


class DepthLimiter:
    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.model = gensim.models.KeyedVectors.load_word2vec_format('data/input/glove.840B.300d.bin', binary=True)
        self.model.init_sims(replace=True)

    def __list_limited_depth_words(self, question, depth_limit):
        sent = next(self.nlp(question).sents)

        words = []
        q = [sent.root]
        for depth in range(depth_limit):
            next_q = []
            for word in q:
                words.append(word)
                if depth + 1 < depth_limit:
                    next_q.extend(list(word.children))
            q = next_q
        return words

    def calculate_row_feature(self, q1, q2):
        depth_limit = 3

        limit_q1 = self.__list_limited_depth_words(q1, depth_limit)
        limit_q2 = self.__list_limited_depth_words(q2, depth_limit)

        filtered_q1 = [word.text for word in limit_q1 if word.text in self.model]
        filtered_q2 = [word.text for word in limit_q2 if word.text in self.model]

        if len(filtered_q1) == 0 or len(filtered_q2) == 0:
            return 0

        return self.model.n_similarity(filtered_q1, filtered_q2)


limiter = None


def create_feature(q1, q2):
    return limiter.calculate_row_feature(q1, q2)


def create_features(data_path):
    data = pd.read_csv(data_path)

    df = pd.DataFrame()
    df['dep_depth_limit'] = Parallel(n_jobs=-1, verbose=5)(
        delayed(create_feature)(q1, q2)
        for q1, q2 in zip(data.question1.astype(str), data.question2.astype(str))
    )
    df.to_csv(feature_output_file(data_path), index=False, float_format='%.5f')


def create_features_files(train_path, test_path):
    if os.path.exists(feature_output_file(train_path)) and os.path.exists(feature_output_file(test_path)):
        print('File exists {}.'.format(feature_output_file(train_path)) + ", " + feature_output_file(test_path))
        return

    global limiter
    limiter = DepthLimiter()

    print('Creating feature for train')
    create_features(train_path)

    print('Creating feature for test')
    create_features(test_path)

def main():
    options = common_feature_parser().parse_args()
    train_path = os.path.join(options.data_prefix, 'train.csv')
    test_path = os.path.join(options.data_prefix, 'test.csv')

    create_features_files(train_path, test_path)


if __name__ == "__main__":
    main()
