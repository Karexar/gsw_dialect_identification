"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import argparse
import os
import time

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

from word_representations.word_embeddings.word_embeddings_model import WordEmbeddingsModel


class ProgressLogger(CallbackAny2Vec):
    """Callback to show progress and training parameters"""

    def __init__(self):
        self.epoch = 1
        self.epoch_start_time = 0
        self.last_training_loss = 0

    def on_epoch_begin(self, model):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss() - self.last_training_loss
        self.last_training_loss = model.get_latest_training_loss()

        print('-' * 70)
        print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | '
              .format(self.epoch, (time.time() - self.epoch_start_time), loss))
        print('-' * 70)

        self.epoch += 1


class Word2VecModel(WordEmbeddingsModel):

    def __init__(self, model_path, model_name, load_from_disk=False):
        WordEmbeddingsModel.__init__(self, model_path, model_name)

        self.word_vectors = None
        self.vectors_path = os.path.join(model_path, f'{model_name}.embed')

        if load_from_disk:
            assert os.path.exists(self.vectors_path)
            print(f'Loading {self.vectors_path} from disk')
            self.word_vectors = KeyedVectors.load(self.vectors_path, mmap='r')
        else:
            # when creating the directory don't include the model_name
            # which is already appended to the passed model_path in self.model_path
            if not os.path.exists(model_path):
                print(f'The directory {model_path} does not exist and will be created')
                os.makedirs(model_path, exist_ok=True)

    def train_model(self, data, embeddings_size, context_size=5, min_count=0, negative_samples=10,
                    training_algorithm=1, workers=5, epochs=20, seed=1111):

        assert len(data) != 0
        print(f'The passed data consists of {len(data)} sentences.')

        print('Training word2vec model')
        model = Word2Vec(
            sentences=data,  # dataset represented as a list of lists
            size=embeddings_size,
            window=context_size,
            min_count=min_count,
            negative=negative_samples,
            sg=training_algorithm,  # 1 is for skip-gram and 0 for CBOW
            seed=seed,
            workers=workers,
            compute_loss=True,
            iter=epochs,
            callbacks=[ProgressLogger()]
        )

        print(f'The vocabulary size of the trained model is {len(model.wv.vocab)} unique tokens')

        model.save(self.model_path)
        self.word_vectors = model.wv
        self.word_vectors.save(self.vectors_path)

    def get_token_embedding(self, token):
        if self.word_vectors:
            if token not in self.word_vectors:
                return None

            return list(self.word_vectors[token])
        else:
            # Load the embedding vectors the first time this method is called
            assert os.path.exists(self.vectors_path)
            print(f'Loading {self.vectors_path}')
            self.word_vectors = KeyedVectors.load(self.vectors_path, mmap='r')
            self.get_token_embedding(token)

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            parser = argparse.ArgumentParser(description='Train and use a Word2Vec word embeddings model')

        group = parser.add_argument_group('Word2Vec')
        group.add_argument('--context-size', dest='context_size', default=5, type=int,
                           help='The size of the context window for training the Word2Vec model')
        group.add_argument('--min-count', dest='min_count', default=5, type=int,
                           help='The threshold for the number of times a word appears in the dataset in order to be'
                                'taken into consideration for the word embeddings training process. Tokens that appear'
                                'less than <min_count> number of times are not considered.')
        group.add_argument('--neg-samples', dest='neg_samples', default=10, type=int,
                           help='The number of negative samples to generate from the dataset')
        group.add_argument('--seed-embed', dest='seed_embed', default=1111, type=int,
                           help='Seed to initialise the random generator')

        return parser


if __name__ == '__main__':
    pass
