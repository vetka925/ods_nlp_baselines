import numpy as np
import scipy.sparse as sp
from textvec.vectorizers import TforVectorizer, TfIcfVectorizer
from scipy.sparse import hstack


class MyTficf(TfIcfVectorizer):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        samples = []
        self.number_of_classes = len(np.unique(y))
        for val in range(self.number_of_classes):
            class_mask = sp.spdiags(y == val, 0, n_samples, n_samples)
            samples.append(np.bincount(
                (class_mask * X).indices, minlength=n_features))
        samples = np.array(samples)
        self.corpus_occurence = np.sum(samples != 0, axis=0)
        self.corpus_occurence = np.where(self.corpus_occurence == 0, 1, self.corpus_occurence)
        self.k = np.log2(1 + (self.number_of_classes / self.corpus_occurence ))
        self._n_features = n_features
        return self

class Vectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class SupervisedVectorizer(Vectorizer):
    def __init__(self, vectorizer, labels):
        super().__init__(vectorizer)
        self.labels = labels

    def fit(self, X):
        if self.labels.nunique() > 2:
            self.supervised = MyTficf(sublinear_tf=True)
        else:
            _labels = self.labels
            self.supervised = TforVectorizer()
        self.supervised.fit(self.vectorizer.transform(X), self.labels)

    def transform(self, X):
        return self.supervised.transform(self.vectorizer.transform(X))


class StackedVectorizer:
    def __init__(self, char_vectorizer, word_vectorizer):
        self.char_vectorizer = char_vectorizer
        self.word_vectorizer = word_vectorizer

    def fit_transform(self, X):
        self.char_vectorizer.fit(X)
        char_features = self.char_vectorizer.transform(X)
        self.word_vectorizer.fit(X)
        word_vectorizer = self.word_vectorizer.transform(X)
        features = hstack([char_features, word_vectorizer])
        return features

    def transform(self, X):
        char_features = self.char_vectorizer.transform(X)
        word_vectorizer = self.word_vectorizer.transform(X)
        features = hstack([char_features, word_vectorizer])
        return features


class SupervisedStackedVectorizer(StackedVectorizer):
    def __init__(self, char_vectorizer, word_vectorizer, labels):
        super().__init__(char_vectorizer, word_vectorizer)
        self.labels = labels

    def fit(self, X):
        if self.labels.nunique() > 2:
            self.supervised_char = MyTficf(sublinear_tf=True)
            self.supervised_word = MyTficf(sublinear_tf=True)
        else:
            self.supervised_char = TforVectorizer(sublinear_tf=True)
            self.supervised_word = TforVectorizer(sublinear_tf=True)


        self.supervised_char.fit(self.char_vectorizer.transform(X), self.labels)
        self.supervised_word.fit(self.word_vectorizer.transform(X), self.labels)

    def transform(self, X):
        char_features = self.supervised_char.transform(self.char_vectorizer.transform(X))
        word_vectorizer = self.supervised_word.transform(self.word_vectorizer.transform(X))
        features = hstack([char_features, word_vectorizer])
        return features
