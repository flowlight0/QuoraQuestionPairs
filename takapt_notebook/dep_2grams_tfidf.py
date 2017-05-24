import os
from typing import Tuple

import en_core_web_md
import pandas as pd
import numpy as np
import spacy
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer



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


creator = DependencyNgramsCreator(n=2, skip_stopwords=False)


def create_ngrams_list(question):
    return ['_'.join(ngram) for ngram in creator.list_ngrams(question)]


def create_ngrams_lists(questions):
    ngrams_strs = Parallel(n_jobs=-1, verbose=5)(
        delayed(create_ngrams_list)(q) for q in questions
    )
    return ngrams_strs


def main():
    input_path = '../input'
    output_path = '../feature'

    train = pd.read_csv(os.path.join(input_path, 'train.csv'))#[:10000]
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))#[:10000]
    
    train_q1_ngram_lists = create_ngrams_lists(train.question1.astype(str))
    train_q2_ngram_lists = create_ngrams_lists(train.question2.astype(str))
    test_q1_ngram_lists = create_ngrams_lists(test.question1.astype(str))
    test_q2_ngram_lists = create_ngrams_lists(test.question2.astype(str))
    
    vectorizer = TfidfVectorizer(tokenizer=lambda a: a, lowercase=False, min_df=10, max_df=0.5)
    vectorizer.fit(train_q1_ngram_lists + train_q2_ngram_lists + test_q1_ngram_lists + test_q2_ngram_lists)

    print('train')
    train_q1_tfidf = vectorizer.transform(train_q1_ngram_lists)
    train_q2_tfidf = vectorizer.transform(train_q2_ngram_lists)
    train_feature = pd.DataFrame()
    train_feature['id'] = train.id
    train_feature['dep_2grams_sum_tfidf_q1'] = train_q1_tfidf.sum(axis=1)
    train_feature['dep_2grams_sum_tfidf_q2'] = train_q2_tfidf.sum(axis=1)
    train_feature['dep_2grams_tfidf_cosine'] = np.array(train_q1_tfidf.multiply(train_q2_tfidf).sum(axis=1)).reshape(-1)
    train_feature.to_pickle(os.path.join(output_path, 'train_dep_2grams_tfidf.pkl'))

    print('test')
    test_q1_tfidf = vectorizer.transform(test_q1_ngram_lists)
    test_q2_tfidf = vectorizer.transform(test_q2_ngram_lists)
    test_feature = pd.DataFrame()
    test_feature['test_id'] = test.test_id
    test_feature['dep_2grams_sum_tfidf_q1'] = test_q1_tfidf.sum(axis=1)
    test_feature['dep_2grams_sum_tfidf_q2'] = test_q2_tfidf.sum(axis=1)
    test_feature['dep_2grams_tfidf_cosine'] = np.array(test_q1_tfidf.multiply(test_q2_tfidf).sum(axis=1)).reshape(-1)
    test_feature.to_pickle(os.path.join(output_path, 'test_dep_2grams_tfidf.pkl'))

if __name__ == "__main__":
    main()
