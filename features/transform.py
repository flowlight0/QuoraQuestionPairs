import os

import nltk
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

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


def sklearn_lda(data_file, n_topics):
    cache_file = data_file + '.lda.{}'.format(n_topics)
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0,
                                    n_jobs=-1)


if __name__ == "__main__":
    nltk_tokenize('../data/input/sample_train.csv')
