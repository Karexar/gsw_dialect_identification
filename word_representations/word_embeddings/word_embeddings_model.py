"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import argparse
import os
from abc import ABC, abstractmethod

import numpy as np


class WordEmbeddingsModel(ABC):

    def __init__(self, model_path, model_name):
        self.model_path = os.path.join(model_path, f'{model_name}.model')

    @abstractmethod
    def get_token_embedding(self, token):
        pass

    def get_batch_embeddings(self, sentence):
        sentence_embeddings = []
        for token in sentence:
            token_embedding = self.get_token_embedding(token)
            if token_embedding is not None:
                sentence_embeddings.append(token_embedding)

        return sentence_embeddings

    def get_vocabulary_embeddings(self, vocab, embeddings_size):
        init_range = 0.1
        num_tokens = len(vocab)
        weights_matrix = np.zeros((num_tokens, embeddings_size))
        print('Initialising embeddings layer matrix with pre-trained embeddings')

        num_oov_tokens = 0
        for idx, token in enumerate(vocab):
            token_embedding = self.get_token_embedding(token)
            if token_embedding is None:
                # if there is no embedding for the given token, return a random representation
                weights_matrix[idx] = np.random.uniform(-init_range, init_range, (1, embeddings_size))
                num_oov_tokens += 1
            else:
                weights_matrix[idx] = np.array(token_embedding)

        print(f'{num_oov_tokens} tokens do not have a representation in the pre-trained model')

        return weights_matrix

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            parser = argparse.ArgumentParser(description='Train and use a word embeddings model')

        group = parser.add_argument_group('Common word embeddings arguments')
        group.add_argument('--model-path-embed', dest='model_path_embed',
                           help='The base directory in which the model should be saved')
        group.add_argument('--model-name-embed', dest='model_name_embed',
                           help='Filename of the trained word embeddings model. This will result in '
                                'the file <model_name>.model being generated in the <model_path> directory.')
        group.add_argument('--train-embed', dest='is_train_embed', action='store_true', default=False,
                           help='Train a word embeddings model')
        group.add_argument('--embed-size', dest='embed_size', default=300, type=int,
                           help='The dimension of the word representation vector')
        group.add_argument('--epochs-embed', dest='epochs_embed', default=20, type=int,
                           help='The number of epochs to train the model')
        group.add_argument('--workers-embed', dest='workers_embed', default=4, type=int,
                           help='Number of jobs to use for the embeddings model training')
