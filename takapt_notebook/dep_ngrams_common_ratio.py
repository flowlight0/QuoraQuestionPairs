import os
from typing import Tuple

import en_core_web_md
import pandas as pd
import spacy
from joblib import Parallel, delayed
from nltk.corpus import stopwords


class DependencyNgramsCreator:
    def __init__(self):
        self.nlp = en_core_web_md.load()
        self.stopwords = set(stopwords.words("english"))

    def list_ngrams(self, text: str, n, skip_stopwords):
        ngrams = set()
        for word in self.nlp(text):
            ngram = self.create_ngram(word, n, skip_stopwords)
            if len(ngram) == n:
                ngrams.add(ngram)
        return ngrams

    def create_ngram(self, word: spacy.tokens.token.Token, n, skip_stopwords) -> Tuple[str]:
        words = []
        while len(words) < n:
            if not skip_stopwords or not (word.text in self.stopwords or word.lemma_ in self.stopwords):
                words.append(word.lemma_)

            if word.dep_ == 'ROOT':
                break
            word = word.head

        if len(words) < n and word.head == word:
            words.append('$ROOT')

        return tuple(reversed(words))


creator = DependencyNgramsCreator()


def create_common_ngrams_ratio_feature(q1, q2, n, skip_stopwords):
    q1_ngrams = creator.list_ngrams(q1, n, skip_stopwords)
    q2_ngrams = creator.list_ngrams(q2, n, skip_stopwords)
    if not q1_ngrams or not q2_ngrams:
        return 0

    common = q1_ngrams & q2_ngrams
    union = q1_ngrams | q2_ngrams
    return len(common) / len(union)


def create_common_ngrams_ratio_features(data, n, skip_stopwords):
    return Parallel(n_jobs=-1, verbose=5)(
        delayed(create_common_ngrams_ratio_feature)(str(row['question1']), str(row['question2']), n, skip_stopwords)
        for index, row in data.iterrows()
    )


def create_df(data, key_column, feature_name, n, skip_stopwords):
    features = create_common_ngrams_ratio_features(data, n, skip_stopwords)
    df = pd.DataFrame()
    df[key_column] = data[key_column]
    df[feature_name] = features
    return df


def main():
    input_path = '../input'
    output_path = '../feature'
    
    train = pd.read_csv(os.path.join(input_path, 'train.csv'))#[:100]
    train_feature = create_df(train, 'id', 'dep_2grams_common_ratio_stop', n=2, skip_stopwords=True)
    train_feature.to_pickle(os.path.join(output_path, 'train_{}.pkl'.format('dep_2grams_common_ratio_stop')))
    
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))#[:100]
    test_feature = create_df(test, 'test_id', 'dep_2grams_common_ratio_stop', n=2, skip_stopwords=True)
    test_feature.to_pickle(os.path.join(output_path, 'test_{}.pkl'.format('dep_2grams_common_ratio_stop')))

if __name__ == "__main__":
    main()
