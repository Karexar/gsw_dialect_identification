"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import argparse
import os

import fasttext

from word_representations.word_embeddings.word_embeddings_model import WordEmbeddingsModel


class FastTextModel(WordEmbeddingsModel):

    def __init__(self, model_path, model_name, load_from_disk=False):
        WordEmbeddingsModel.__init__(self, model_path, model_name)

        self.word_vectors = None

        if load_from_disk:
            assert os.path.exists(self.model_path)
            print(f'Loading {self.model_path} from disk')
            # self.word_vectors = KeyedVectors.load(self.vectors_path, mmap='r')
            self.word_vectors = fasttext.load_model(self.model_path)
        else:
            # when creating the directory don't include the model_name
            # which is already appended to the passed model_path in self.model_path
            if not os.path.exists(model_path):
                print(f'The directory {model_path} does not exist and will be created')
                os.makedirs(model_path, exist_ok=True)

    def train_model(self, data, embeddings_size, context_size=5, learning_rate=0.05, min_count=5, negative_samples=5,
                    training_algorithm='skipgram', min_char_ngram=3, max_char_ngram=6, num_buckets=2000000,
                    loss_function='ns', workers=4, epochs=20):

        assert os.path.exists(data)
        print(f'Using the sentences from {data} to train the FastText model.')

        print('Training FastText model')
        model = fasttext.train_unsupervised(
            input=data,  # the path to the dataset file
            dim=embeddings_size,
            ws=context_size,
            min_count=min_count,
            minn=min_char_ngram,  # the smallest character n-gram to take as a subword
            maxn=max_char_ngram,  # the largest character n-gram to take as a subword
            neg=negative_samples,
            model=training_algorithm,  # the default is skip-gram
            lr=learning_rate,
            loss=loss_function,
            bucket=num_buckets,
            thread=workers,
            epoch=epochs
        )

        print(f'The vocabulary size of the trained model is {len(model.words)} unique tokens')

        model.save_model(self.model_path)
        self.word_vectors = model

    def get_token_embedding(self, token):
        if self.word_vectors:
            # Even words that don't appear in the training set
            # should have a suitable representation with FastText
            return self.word_vectors.get_word_vector(token)
        else:
            # Load the embedding vectors the first time this method is called
            assert os.path.exists(self.model_path)
            print(f'Loading {self.model_path}')
            self.word_vectors = fasttext.load_model(self.model_path)
            self.get_token_embedding(token)

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            parser = argparse.ArgumentParser(description='Train and use a FastText word embeddings model')

        group = parser.add_argument_group('FastText')
        group.add_argument('--data-file', dest='data_file',
                           help='The path of the data file which contains the word embeddings training sentences')
        group.add_argument('--context-size', dest='context_size', default=5, type=int,
                           help='The size of the context window for training the Word2Vec model')
        group.add_argument('--min-count', dest='min_count', default=5, type=int,
                           help='The threshold for the number of times a word appears in the dataset in order to be'
                                'taken into consideration for the word embeddings training process. Tokens that appear'
                                'less than <min_count> number of times are not considered.')
        group.add_argument('--min-char-ngram', dest='min_char_ngram', default=3, type=int,
                           help='The smallest character n-gram to take as a subword')
        group.add_argument('--max-char-ngram', dest='max_char_ngram', default=6, type=int,
                           help='The largest character n-gram to take as a subword')
        group.add_argument('--neg-samples', dest='neg_samples', default=5, type=int,
                           help='The number of negative samples to generate from the dataset')
        group.add_argument('--num-buckets', dest='num_buckets', default=2000000, type=int,
                           help='The number of buckets to use in the hashing of the subwords')
        group.add_argument('--lr-embed', dest='lr_embed', default=0.05, type=float,
                           help='The learning rate for the FastText model')
        group.add_argument('--loss-embed', dest='loss_embed', default='ns', choices=['ns', 'hs', 'softmax', 'ova'],
                           help='The loss function to use to train the model')

        return parser


if __name__ == '__main__':
    pass
