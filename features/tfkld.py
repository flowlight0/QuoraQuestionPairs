import math

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted


class TfKLdVectorizer:
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64,
                 norm='l2', alpha=0.5, symmetric=True, divergence='kl'):
        self._count_vectorizer = CountVectorizer(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)
        self._tfkld_transformer = TfKLdTransformer(norm=norm, alpha=alpha, symmetric=symmetric, divergence=divergence)

    def fit(self, raw_documents1, raw_documents2, y):
        self._count_vectorizer.fit(raw_documents=raw_documents1 + raw_documents2)
        self._tfkld_transformer.fit(self._count_vectorizer.transform(raw_documents1),
                                    self._count_vectorizer.transform(raw_documents2), y)
        return self

    def transform(self, raw_documents):
        check_is_fitted(self, '_count_vectorizer')
        return self._tfkld_transformer.transform(self._count_vectorizer.transform(raw_documents))


class TfKLdTransformer:
    def __init__(self, norm='l2', alpha=0.5, symmetric=True, divergence='kl'):
        self.weight = None
        self.norm = norm
        self.alpha = alpha
        self.symmetric = symmetric
        self.divergence = divergence

    def fit(self, X1, X2, y=None):
        if y is None:
            raise ValueError("y must not be None")

        if X1.shape != X2.shape:
            raise ValueError("X1 and X2 have different shape. X1 %s, and X2 %s." % (X1.shape, X2.shape))

        if X1.shape[0] != len(y):
            raise ValueError("X1, X2 and y have different examples. X1 %d, X2 %d, and y %d." % (X1.shape[0], X2.shape[0], len(y)))

        if not isinstance(X1, scipy.sparse.csc.csc_matrix):
            X1 = scipy.sparse.csc_matrix(X1)

        if not isinstance(X2, scipy.sparse.csc.csc_matrix):
            X2 = scipy.sparse.csc_matrix(X2)

        if self.symmetric:
            self.weight = (self._fit(X1, X2, y) + self._fit(X2, X1, y)) / 2
        else:
            self.weight = self._fit(X1, X2, y)
        return self

    def _fit(self, X1: scipy.sparse.csc.csc_matrix, X2: scipy.sparse.csc.csc_matrix, y):
        w = np.zeros((X1.shape[1],))
        for i in range(X1.shape[1]):
            print(i, X1.shape[1])
            rows, cols = X1.getcol(i).nonzero()
            tmp_x = np.array(X2.getcol(i)[rows, cols]).flatten()
            tmp_y = y[rows]
            w[i] = self._weight(tmp_x[tmp_y == 1], tmp_x[tmp_y == 0])
        return w

    def _weight(self, dist_p, dist_q):
        p = ((dist_p > 0).sum() + self.alpha) / (dist_p.shape[0] + self.alpha * 2)
        q = ((dist_q > 0).sum() + self.alpha) / (dist_q.shape[0] + self.alpha * 2)
        if self.divergence == 'kl':
            return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        elif self.divergence == 'js':
            r = (p + q) / 2
            return (p * math.log(p / r) + (1 - p) * math.log((1 - p) / (1 - r)) +
                    q * math.log(q / r) + (1 - q) * math.log((1 - q) / (1 - r))) / 2

    def transform(self, X):
        weight_diag = scipy.sparse.diags(self.weight.ravel(), shape=(X.shape[1], X.shape[1]), dtype=np.float32)
        X = X * weight_diag
        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        return X


def main():
    q1 = ['abc', 'ab']
    q2 = ['ab', 'bc']
    qs = q1 + q2
    y = np.array([1, 0])
    vectorizer = TfKLdVectorizer(analyzer='char', alpha=0.01, divergence='kl')
    vectorizer.fit(q1, q2, y)
    print(vectorizer.transform(qs).toarray())

if __name__ == "__main__":
    main()
