import os

import gensim
import joblib
import nltk
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

from features.utils import get_stop_words


def nltk_tokenize(data_file):
    cache_file = data_file + '.tokenized'
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    data = pd.read_csv(data_file)
    data['question1'] = data['question1'].apply(lambda s: " ".join(nltk.word_tokenize(str(s).lower()))).values
    data['question2'] = data['question2'].apply(lambda s: " ".join(nltk.word_tokenize(str(s).lower()))).values
    data.to_csv(cache_file, index=False)
    return data


def stemming_words(words: list, stopwords, stemmer: nltk.stem.PorterStemmer):
    stemmed_words = []
    for word in words:
        if word not in stopwords:
            try:
                stemmed_words.append(stemmer.stem(word))
            except IndexError:
                stemmed_words.append(word)
    return stemmed_words


def nltk_stemming(data_file):
    stemmer = nltk.stem.PorterStemmer()
    swords = get_stop_words()
    cache_file = data_file + '.stemmed'
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    data = nltk_tokenize(data_file)
    data['question1'] = data['question1'].apply(
        lambda s: " ".join(stemming_words(str(s).split(), stopwords=swords, stemmer=stemmer))
    ).values
    data['question2'] = data['question2'].apply(
        lambda s: " ".join(stemming_words(str(s).split(), stopwords=swords, stemmer=stemmer))
    ).values
    data.to_csv(cache_file, index=False)
    return data


def nltk_pos_tag(data_file):
    cache_file = data_file + '.pos'
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    data = pd.read_csv(data_file)
    data['question1'] = data['question1'].apply(lambda s: nltk.pos_tag(nltk.word_tokenize(str(s)))).values
    data['question2'] = data['question2'].apply(lambda s: nltk.pos_tag(nltk.word_tokenize(str(s)))).values
    data.to_csv(cache_file, index=False)
    return data


def sentence2vec(data_file):
    cache_file = data_file + '.sentence2vec'
    if os.path.exists(cache_file):
        return joblib.load(cache_file)

    model = gensim.models.KeyedVectors.load_word2vec_format('data/input/GoogleNews-vectors-negative300.bin', binary=True)
    data = nltk_tokenize(data_file)
    question1_vectors = np.zeros((data.shape[0], 300))  # Google's w2v has 300 elements in each feature
    question2_vectors = np.zeros((data.shape[0], 300))  # Google's w2v has 300 elements in each feature
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = _sentence2vec(q, model=model)

    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = _sentence2vec(q, model=model)

    joblib.dump((question1_vectors, question2_vectors), cache_file)
    return question1_vectors, question2_vectors


def _sentence2vec(s: str, model):
    swords = get_stop_words()
    words = str(s).lower().split()
    words = [w for w in words if w not in swords]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


if __name__ == "__main__":
    nltk_tokenize('../data/input/sample_train.csv')
