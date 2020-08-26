from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from joblib import dump, load # To save classifiers on disk
from typing import List

class Vectorizer:
    """Vectorizer for written sentences, using words and characters n-grams

    Attributes
    ----------
    vectorizers : List[sklearn.feature_extraction.text.TfidfVectorizer]
        A list of TfIdf vectorizers. Each one taking care of one feature type
        (e.g. character 2-gram, word unigram...)

    Methods
    -------
    train(sentences, word_ngrams, char_ngrams)
        Create the language model using a training set of sentences
    vectorize(sentences, vect_idx)
        Vectorize a list of sentences using word and character n-grams
    """

    def __init__(self):
        self.vectorizers = []

    def train(self,
              sentences: List[str],
              word_ngrams=[1],
              char_ngrams=[]):
        """Vectorize a list of sentences using word and character n-grams

        One vectorizer per feature type (n) is used. This is because we may
        want to use either one classifier with a full vector containing all
        feature types (i.e. all n for word and char n-grams),
        or several classifiers with partial vector representing one feature type.
        If word and character n-grams are used together, two different
        vectorizers are used, and the resulting matrices are concatenated.

        Parameters
        ----------
        sentences : List[str]
            A list of sentences to train the language model
        word_ngram : List[int]
            A list of int representing the n for word n-grams
        char_ngram : List[int]
            A list of int representing the n for character n-grams

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            A sparse matrix representing the vectorization of the sentences
        """

        for n in word_ngrams:
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(n,n))
            vectorizer.fit(sentences)
            self.vectorizers.append(vectorizer)

        for n in char_ngrams:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n,n))
            vectorizer.fit(sentences)
            self.vectorizers.append(vectorizer)

    def vectorize(self,
                  sentences,
                  vect_idx=-1):
        """Vectorize a list of sentences using word and character n-grams

        In the current implementation, the output is a single matrices containing
        the full vector (i.e. all feature types) for all sentences.

        Parameters
        ----------
        sentences : List[str]
            A list of sentences to vectorize
        vect_idx : int
            The idx of the vectorizer to use. If -1 is given, then all
            vectorizers are used and the resulting vectors are concatenated.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            A sparse matrix representing the vectorization of the sentences
        """

        if vect_idx != -1:
            return self.vectorizers[vect_idx].transform(sentences)
        else:
            final_vec = None
            for vectorizer in self.vectorizers:
                partial_vec = vectorizer.transform(sentences)
                final_vec = sparse.hstack([final_vec, partial_vec])

            return final_vec
