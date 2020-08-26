"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import os

import torch
from torch import nn


class LSTM(nn.Module):
    """Contains the architecture, initialisation and training steps for an LSTM model"""

    def __init__(self, vocab_size, embedding_dim, num_hidden, num_layers, dropout, num_labels, embeddings_matrix=None):
        super(LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_labels = num_labels
        self.embeddings_matrix = embeddings_matrix

        # Create model layers
        self.word_embeddings_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        self.lstm_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.num_hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            in_features=self.num_hidden,
            out_features=self.num_labels
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        self.init_weights()
        self.init_embeddings()

    def init_weights(self):
        """Initialises the embeddings and the output layers of the model"""
        init_range = 0.1
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-init_range, init_range)

    def init_embeddings(self):
        if self.embeddings_matrix is None:
            init_range = 0.1
            self.word_embeddings_layer.weight.data.uniform_(-init_range, init_range)
        else:
            self.word_embeddings_layer.weight.data.copy_(torch.from_numpy(self.embeddings_matrix))

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # The batch size can be either for training or for evaluation,
        # therefore it is passed as a method argument
        # One item of the tuple is for the memory and the other for the hidden state
        return (
            weight.new_zeros(self.num_layers, batch_size, self.num_hidden),
            weight.new_zeros(self.num_layers, batch_size, self.num_hidden)
        )

    def forward(self, inputs, hidden):
        embeddings = self.dropout_layer(self.word_embeddings_layer(inputs))
        output, hidden = self.lstm_layer(embeddings, hidden)
        output = self.dropout_layer(output)
        # only take the last output of the LSTM, which is the 2nd dimension
        output = output[:, -1]
        logits = self.output_layer(output)
        return logits, hidden, output

    @staticmethod
    def repackage_hidden(hidden):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(LSTM.repackage_hidden(var) for var in hidden)

    def export_model(self, path):
        print(f'Export model to: {path}')

        model_checkpoint = {
            'parameters': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'num_hidden': self.num_hidden,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'num_labels': self.num_labels,
                'embeddings_matrix': self.embeddings_matrix
            },
            'state_dict': self.state_dict()
        }

        torch.save(model_checkpoint, f'{path}')

    @staticmethod
    def import_model(path, device):
        assert os.path.exists(path)
        print(f'Import model from: {path}')

        model_checkpoint = torch.load(path, map_location=device)
        model = LSTM(
            model_checkpoint['parameters']['vocab_size'],
            model_checkpoint['parameters']['embedding_dim'],
            model_checkpoint['parameters']['num_hidden'],
            model_checkpoint['parameters']['num_layers'],
            model_checkpoint['parameters']['dropout'],
            model_checkpoint['parameters']['num_labels'],
            model_checkpoint['parameters']['embeddings_matrix']
        ).to(device)

        model.load_state_dict(model_checkpoint['state_dict'])
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.lstm_layer.flatten_parameters()

        return model
