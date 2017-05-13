import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.feature_creation import StandardS3LocalCachedCreator
from features.utils import get_stop_words


class F0001Creator(StandardS3LocalCachedCreator):
    def __init__(self, base_dir, feature_name, bucket_name=None, **args):
        self.weights = None
        super().__init__(base_dir, feature_name, bucket_name=bucket_name, **args)

    def prepare_without_cache_check(self, train_data, test_data):
        df_train = pd.read_csv(train_data)
        df_test = pd.read_csv(test_data)
        train_qs = pd.Series(df_train['question1'].tolist() +
                             df_train['question2'].tolist() +
                             df_test['question1'].tolist() +
                             df_test['question2'].tolist()).astype(str)
        self.weights = self.get_weights(train_qs)

    @staticmethod
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

    def calculate_feature(self, data):
        data = pd.read_csv(data)
        values = []
        swords = get_stop_words()
        for i, row in tqdm(data.iterrows()):
            values.append(self._calculate_feature(row=row, swords=swords))
        data[self.feature_name] = values
        return data[[self.feature_name]]

    def _calculate_feature(self, row, swords):
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

        shared_weights = [self.weights.get(w, 0) for w in q1words.keys() if w in q2words] + \
                         [self.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]
        den = np.sum(total_weights)
        return np.sum(shared_weights) / den if den > 0 else 0


if __name__ == "__main__":
    base_dir = "/Users/takanori/Study/machine-learning/Kaggle/QuoraQuestionPairs"
    feature_name = os.path.splitext(os.path.basename(__file__))[0]
    train_data = os.path.join(base_dir, 'data/input/train.csv')
    test_data = os.path.join(base_dir, 'data/input/test.csv')
    creator = F0001Creator(base_dir=base_dir, feature_name=feature_name, bucket_name="flowlight-quora-question-pairs",
                           train_data=train_data, test_data=test_data)
    creator.create_feature(train_data)
    creator.create_feature(test_data)
