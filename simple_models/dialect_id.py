from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load # To save classifiers on disk
from naive_bayes.vectorizer import *
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from statistics import mean
import numpy as np
import os

class Dialect_identification:
    """Predict the dialect from swiss-german written sentences

    Attributes
    ----------
    clf : sklearn classifier
        The classifier used to predict dialects
    vectorizer :
        The vectorizer used for sentences

    Methods
    -------
    train_vectorizer(sentences, out_path, word_ngrams, char_ngrams)
        Train a language model to vectorize sentences and save it on disk
    train(path, sentences, labels, out_path, vect_idx, clf)
        Train the prediction model
    predict(sentences)
        Predict the swiss-german dialect from a list of sentences
    """

    def __init__(self,
                 vec_path=None,
                 clf_path=None,
                 clfs_dir=None):
        """Load an existing model if existing

        Parameters
        ----------
        vec_path : str
            The path to an existing vectorizer (extension is .joblib)
        clf_path : str
            The path to an existing sklearn classifier (extension is .joblib)
        clfs_dir : str
            The directory containing an ensemble of classifier (.joblib). The
            prediction will be the mean of these classifier predictions
        """
        if vec_path:
            print("Loading " + vec_path)
            self.vectorizer = load(vec_path)
        else:
            self.vectorizer = Vectorizer()

        self.clf = None
        if clf_path:
            print("Loading " + clf_path)
            load(clf_path)

        self.clfs = []
        if clfs_dir:
            print("Loading classifier ensemble")
            names = [x for x in os.listdir(clfs_dir) if x.endswith(".joblib")]
            for name in names:
                path = os.path.join(clfs_dir, name)
                print("Loading " + path)
                self.clfs.append(load(path))

    def train_vectorizer(self,
                         sentences,
                         out_path,
                         word_ngrams=[1],
                         char_ngrams=[]):
        """Train a language model to vectorize sentences and save it on disk

        Parameters
        ----------
        sentences : List[str]
            A list of sentences to train the language model
        out_path : str
            The path where to save the language model, extension is .joblib
        word_ngram : List[int]
            A list of int representing the n for word n-grams
        char_ngram : List[int]
            A list of int representing the n for character n-grams

        Returns
        -------
        None
        """
        self.vectorizer.train(sentences, word_ngrams, char_ngrams)
        print("Saving vectorizer to " + out_path)
        dump(self.vectorizer, out_path)

    def train(self,
              sentences,
              labels,
              out_path,
              vect_idx=-1,
              clf=MultinomialNB(alpha=0.01, fit_prior=False)):
        """Train the prediction model

        Parameters
        ----------
        sentences : List[str]
            A list of sentences to train the model on
        labels : List[str]
            A list of labels to train the model on
        out_path : str
            The path where to save the classifier, extension is .joblib
        vect_idx : int
            The index of the vectorizer to use. If -1, then all vectorizers
            are used and the partial vectors are concatenated.
        clf : sklearn classifier
            The sklearn classifier to use for prediction

        Returns
        -------
        None
        """
        self.clf = clf

        X = self.vectorizer.vectorize(sentences, vect_idx)
        if vect_idx == -1:
            self.clf.fit(X, labels)
            print("Saving vectorizer to " + out_path)
            dump(self.clf, out_path)
        else:
            clf.fit(X, labels)
            print("Saving vectorizer to " + out_path)
            dump(clf, out_path)
            self.clfs.append(clf)

    def k_fold(self,
               sentences,
               labels,
               k,
               clf=MultinomialNB(alpha=0.01, fit_prior=False)):
        """Train the prediction model

        Parameters
        ----------
        sentences : List[str]
            A list of sentences to train the model on
        labels : List[str]
            A list of labels to train the model on
        k : int
            Number of folds
        clf : sklearn classifier
            The sklearn classifier to use for prediction

        Returns
        -------
        None
        """

        self.clf = clf
        k_fold = KFold(n_splits=k, shuffle=True, random_state=0)
        X = self.vectorizer.vectorize(sentences, -1)
        # scoring : e.g. f1_weighted or accuracy
        scores = cross_validate(self.clf, X, labels, cv=k_fold, n_jobs=1, scoring='accuracy')
        return scores

    def predict(self,
                sentences):
        """Predict the swiss-german dialect from a list of sentences

        Parameters
        ----------
        sentences : List[str]
            A list of swiss-german sentences

        Returns
        -------
        List[str]
            A list of swiss-german dialect predictions
        """
        if len(self.clfs) == 0:
            return self.clf.predict(self.vectorizer.vectorize(sentences))
        else:
            vect_matrix = self.vectorizer.vectorize(sentences, 0)
            res = self.clfs[0].predict_proba(vect_matrix)
            for i in range(1, len(self.clfs)):
                vect_matrix = self.vectorizer.vectorize(sentences, i)
                pred = self.clfs[i].predict_proba(vect_matrix)
                res = res + pred
            idx_max = np.argmax(res, axis=1)

            return [self.clfs[0].classes_[x] for x in idx_max]
