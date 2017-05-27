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


class DependencyNgramsCreator:
    def __init__(self, n, skip_stopwords):
        self.n = n
        self.skip_stopwords = skip_stopwords

        self.nlp = en_core_web_md.load()
        self.stopwords = set(stopwords.words("english"))

    def list_ngrams(self, text: str):
        ngrams = set()
        for word in self.nlp(text):
            ngram = self.create_ngram(word)
            if len(ngram) == self.n:
                ngrams.add(ngram)
        return ngrams

    def create_ngram(self, word: spacy.tokens.token.Token) -> Tuple[str]:
        words = []
        while len(words) < self.n:
            if not self.skip_stopwords or not (word.text in self.stopwords or word.lemma_ in self.stopwords):
                words.append(word.lemma_)

            if word.dep_ == 'ROOT':
                break
            word = word.head

        if len(words) < self.n and word.head == word:
            words.append('$ROOT')

        return tuple(reversed(words))


creator = None
vectorizer = None


def create_ngrams_list(question):
    return ['_'.join(ngram) for ngram in creator.list_ngrams(question)]


def create_ngrams_lists(questions):
    ngrams_strs = Parallel(n_jobs=-1, verbose=5)(
        delayed(create_ngrams_list)(q) for q in questions
    )
    return ngrams_strs


def create_feature(train_path, test_path, n, skip_stopwords):
    if os.path.exists(feature_output_file(train_path)) and os.path.exists(feature_output_file(test_path)):
        print('File exists {}.'.format(feature_output_file(train_path)) + ", " + feature_output_file(test_path))
        return

    global creator
    global vectorizer
    print('start preprocessing')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    creator = DependencyNgramsCreator(n=n, skip_stopwords=skip_stopwords)

    train_q1_ngram_lists = create_ngrams_lists(train.question1.astype(str))
    train_q2_ngram_lists = create_ngrams_lists(train.question2.astype(str))
    test_q1_ngram_lists = create_ngrams_lists(test.question1.astype(str))
    test_q2_ngram_lists = create_ngrams_lists(test.question2.astype(str))

    vectorizer = TfidfVectorizer(tokenizer=lambda a: a, lowercase=False, min_df=10, max_df=0.5)
    vectorizer.fit(train_q1_ngram_lists + train_q2_ngram_lists + test_q1_ngram_lists + test_q2_ngram_lists)

    print('finish preprocessing')

    print('train')
    train_q1_tfidf = vectorizer.transform(train_q1_ngram_lists)
    train_q2_tfidf = vectorizer.transform(train_q2_ngram_lists)
    train_feature = pd.DataFrame()
    train_feature['dep_2grams_stop_sum_tfidf_q1'] = np.array(train_q1_tfidf.sum(axis=1)).flatten()
    train_feature['dep_2grams_stop_sum_tfidf_q2'] = np.array(train_q2_tfidf.sum(axis=1)).flatten()
    train_feature['dep_2grams_stop_tfidf_cosine'] = np.array(train_q1_tfidf.multiply(train_q2_tfidf).sum(axis=1)).flatten()
    train_feature.to_csv(feature_output_file(train_path), index=False, float_format='%.5f')

    print('test')
    test_q1_tfidf = vectorizer.transform(test_q1_ngram_lists)
    test_q2_tfidf = vectorizer.transform(test_q2_ngram_lists)
    test_feature = pd.DataFrame()
    test_feature['dep_2grams_stop_sum_tfidf_q1'] = np.array(test_q1_tfidf.sum(axis=1)).flatten()
    test_feature['dep_2grams_stop_sum_tfidf_q2'] = np.array(test_q2_tfidf.sum(axis=1)).flatten()
    test_feature['dep_2grams_stop_tfidf_cosine'] = np.array(test_q1_tfidf.multiply(test_q2_tfidf).sum(axis=1)).flatten()
    test_feature.to_csv(feature_output_file(test_path), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    train_path = os.path.join(options.data_prefix, 'train.csv')
    test_path = os.path.join(options.data_prefix, 'test.csv')
    create_feature(train_path, test_path, n=2, skip_stopwords=True)

if __name__ == "__main__":
    main()
