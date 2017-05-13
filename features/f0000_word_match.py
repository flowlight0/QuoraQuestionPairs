import os

import pandas as pd
from tqdm import tqdm

from features.feature_creation import StandardS3LocalCachedCreator
from features.utils import get_stop_words


class F0000Creator(StandardS3LocalCachedCreator):
    def prepare_without_cache_check(self, train_data, test_data):
        pass

    def prepare_with_cache_check(self, **params):
        pass

    def calculate_feature(self, data):
        data = pd.read_csv(data)
        values = []
        swords = get_stop_words()
        for i, row in tqdm(data.iterrows()):
            values.append(self._calculate_feature(row=row, swords=swords))
        data[self.feature_name] = values
        return data[[self.feature_name]]

    @staticmethod
    def _calculate_feature(row, swords):
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
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))


if __name__ == "__main__":
    base_dir = "/Users/takanori/Study/machine-learning/Kaggle/QuoraQuestionPairs"
    feature_name = os.path.splitext(os.path.basename(__file__))[0]
    train_data = os.path.join(base_dir, 'data/input/train.csv')
    test_data = os.path.join(base_dir, 'data/input/test.csv')
    creator = F0000Creator(base_dir=base_dir, feature_name=feature_name, bucket_name="flowlight-quora-question-pairs",
                           train_data=train_data, test_data=test_data)
    creator.create_feature(train_data)
    creator.create_feature(test_data)
