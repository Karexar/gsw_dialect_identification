"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import argparse
import os
import time

from glove import Glove#, Corpus

from word_representations.word_embeddings.word_embeddings_model import WordEmbeddingsModel


class GloveModel(WordEmbeddingsModel):

    def __init__(self, model_path, model_name, load_from_disk=False):
        WordEmbeddingsModel.__init__(self, model_path, model_name)

        self.word_vectors = None
        self.corpus_path = os.path.join(model_path, f'{model_name}.corpus')

        if load_from_disk:
            assert os.path.exists(self.model_path)
            print(f'Loading {self.model_path} from disk')
            self.word_vectors = Glove.load(self.model_path)
        else:
            # when creating the directory don't include the model_name
            # which is already appended to the passed model_path in self.model_path
            if not os.path.exists(model_path):
                print(f'The directory {model_path} does not exist and will be created')
                os.makedirs(model_path, exist_ok=True)

    def train_model(self, data, embeddings_size, context_size=10, learning_rate=0.05, alpha=0.75, max_count=100,
                    max_loss=10.0, dictionary=None, workers=5, epochs=20, seed=1111):

        assert len(data) != 0
        print(f'The passed data consists of {len(data)} sentences.')

        print('Fitting the corpus with the given training sentences')
        start_time = time.time()
        if os.path.exists(self.corpus_path):
            print(f'Loading corpus {self.corpus_path} from disk')
            corpus = Corpus.load(self.corpus_path)
        else:
            corpus = Corpus(dictionary=dictionary)
            # dataset represented as a list of lists
            # the length of the (symmetric) context window used for co-occurrence
            corpus.fit(data, window=context_size)
            corpus.save(self.corpus_path)

        print('| Corpus fit time: {:5.2f}s |'.format(time.time() - start_time))
        print(f'The vocabulary size of the trained model is {len(corpus.dictionary)} unique tokens')
        print(f'The number of collocations is {corpus.matrix.nnz}')

        print('Training GloVe model')
        start_time = time.time()
        model = Glove(
            no_components=embeddings_size,  # number of latent dimensions
            alpha=alpha,
            max_count=max_count,
            max_loss=max_loss,
            learning_rate=learning_rate,
            random_state=seed,
        )

        # fitting to the corpus and adding standard dictionary to the object
        model.fit(corpus.matrix, epochs=epochs, no_threads=workers, verbose=True)
        model.add_dictionary(corpus.dictionary)

        print('| GloVe model training time: {:5.2f}s |'.format(time.time() - start_time))

        model.save(self.model_path)
        self.word_vectors = model

    def get_token_embedding(self, token):
        if self.word_vectors:
            if token not in self.word_vectors.dictionary:
                return None

            token_id = self.word_vectors.dictionary[token]
            return self.word_vectors.word_vectors[token_id]
        else:
            # Load the embedding vectors the first time this method is called
            assert os.path.exists(self.model_path)
            print(f'Loading {self.model_path} from disk')
            self.word_vectors = Glove.load(self.model_path)
            self.get_token_embedding(token)

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            parser = argparse.ArgumentParser(description='Train and use a Glove word embeddings model')

        group = parser.add_argument_group('GloVe')
        group.add_argument('--context-size', dest='context_size', default=5, type=int,
                           help='The size of the context window for training the Glove model')
        group.add_argument('--alpha', dest='alpha', default=0.75, type=float,
                           help='The alpha parameter for the GloVe model')
        group.add_argument('--max-count', dest='max_count', default=100, type=float,
                           help='The max count parameter for the GloVe model')
        group.add_argument('--lr-embed', dest='lr_embed', default=0.05, type=float,
                           help='The learning rate for the GloVe model')
        group.add_argument('--max-loss', dest='max_loss', default=10.0, type=float,
                           help='The learning rate for the GloVe model')
        group.add_argument('--seed-embed', dest='seed_embed', default=1111, type=int,
                           help='Seed to initialise the random generator')

        return parser


if __name__ == '__main__':
    pass
