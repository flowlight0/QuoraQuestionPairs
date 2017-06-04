import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from ffmutil import create_ffm_field_file


class NgramFFMFeatureCreator:
    def __init__(self, ngram):
        self.vectorizer = CountVectorizer(min_df=4, ngram_range=(ngram, ngram))

    def fit(self, qs):
        self.vectorizer.fit(qs)

    def create(self, qs):
        count_vecs = self.vectorizer.transform(qs)
        return [list(count_vec.indices) for count_vec in count_vecs]

#
# def create_field_feature_files(data):
#     pass


def main():
    train = pd.read_csv('../data/input/train.csv.stemmed')
    train_q1 = train.question1.astype(str).tolist()
    train_q2 = train.question1.astype(str).tolist()

    train_qs = train_q1 + train_q2

    for ngram in tqdm(range(1, 6)):
        creator = NgramFFMFeatureCreator(ngram)
        creator.fit(train_qs)

        q1_features = creator.create(train_q1)
        q2_features = creator.create(train_q2)

        create_ffm_field_file(q1_features, 'train', '{}grams_stem_q1'.format(ngram))
        create_ffm_field_file(q2_features, 'train', '{}grams_stem_q2'.format(ngram))


if __name__ == '__main__':
    main()

