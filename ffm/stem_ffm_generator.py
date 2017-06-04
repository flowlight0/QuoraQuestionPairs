import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from joblib import Parallel, delayed
from ffmutil import create_ffm_field_file


class NgramFFMFeatureCreator:
    def __init__(self, ngram):
        self.vectorizer = CountVectorizer(min_df=4, ngram_range=(ngram, ngram))

    def fit(self, qs):
        self.vectorizer.fit(qs)

    def create(self, qs):
        count_vecs = self.vectorizer.transform(qs)
        return [list(count_vec.indices) for count_vec in count_vecs]


def create(ngram, train_qs, train_q1, train_q2, test_q1, test_q2):
    creator = NgramFFMFeatureCreator(ngram)
    creator.fit(train_qs)
    print('{}grams: fit done'.format(ngram))

    train_q1_features = creator.create(train_q1)
    train_q2_features = creator.create(train_q2)
    create_ffm_field_file(train_q1_features, 'train', '{}grams_stem_q1'.format(ngram))
    create_ffm_field_file(train_q2_features, 'train', '{}grams_stem_q2'.format(ngram))
    print('{}grams: train features done'.format(ngram))

    test_q1_features = creator.create(test_q1)
    test_q2_features = creator.create(test_q2)
    create_ffm_field_file(test_q1_features, 'test', '{}grams_stem_q1'.format(ngram))
    create_ffm_field_file(test_q2_features, 'test', '{}grams_stem_q2'.format(ngram))
    print('{}grams: test features done'.format(ngram))


def main():
    train = pd.read_csv('../data/input/train.csv.stemmed')#[:1000]
    train_q1 = train.question1.astype(str).tolist()
    train_q2 = train.question1.astype(str).tolist()
    
    test = pd.read_csv('../data/input/test.csv.stemmed')#[:1000]
    test_q1 = test.question1.astype(str).tolist()
    test_q2 = test.question1.astype(str).tolist()

    train_qs = train_q1 + train_q2

    Parallel(n_jobs=6, verbose=5)(
        delayed(create)(ngram, train_qs, train_q1, train_q2, test_q1, test_q2) for ngram in range(1, 6)
    )


if __name__ == '__main__':
    main()

