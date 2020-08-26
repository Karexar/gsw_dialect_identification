"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import os

import torch
from torchtext import data
from torchtext.data import TabularDataset


class DataManager:

    def __init__(self, path, device, batch_size=30, eval_batch_size=20):
        assert os.path.exists(path), f'The specified directory {path} does not exist'

        self.path = path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        text_field = data.Field(sequential=True,
                                tokenize=DataManager._tokenize_text,
                                use_vocab=True,
                                batch_first=True,
                                include_lengths=True)
        dialect_field = data.LabelField(use_vocab=True,
                                        tensor_type=torch.FloatTensor)

        data_fields = [("text", text_field),
                       ("dialect", dialect_field)]

        self.train_data, self.valid_data, self.test_data = TabularDataset.splits(
            path=self.path,
            train='train.csv',
            validation='dev.csv',
            test='test.csv',
            format='tsv',
            skip_header=False,
            fields=data_fields
        )

        print(f'Training set: {len(self.train_data.examples)}')
        print(f'Validation set: {len(self.valid_data.examples)}')
        print(f'Test set: {len(self.test_data.examples)}')

        #print(self.train_data.examples[0].text, self.train_data.examples[0].dialect)
        #print(self.valid_data.examples[0].text, self.valid_data.examples[0].dialect)
        #print(self.test_data.examples[0].text, self.test_data.examples[0].dialect)

        text_field.build_vocab(self.train_data, self.valid_data, self.test_data)
        dialect_field.build_vocab(self.train_data)

        self.vocab_size = len(text_field.vocab)
        self.num_labels = len(dialect_field.vocab)
        print(f'Vocabulary size: {self.vocab_size}')
        print(f'Number of dialect labels: {self.num_labels}')

        self.idx2word = text_field.vocab.itos
        self.word2idx = text_field.vocab.stoi
        self.idx2dialect = dialect_field.vocab.itos

        # print(dialect_field.vocab.freqs)
        # print(dialect_field.vocab.itos)
        # print(dialect_field.vocab.stoi)

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_sizes=(self.batch_size, self.eval_batch_size, self.eval_batch_size),
            sort_key=lambda x: len(x.text), device=device, repeat=False, shuffle=True)

        # for idx, batch in enumerate(self.train_iter):
        #     text = batch.text[0]  # the other item in the tuple is the length of the mini-batch
        #     target = batch.dialect
        #     print(text.data, target.data)
        #
        #     for token_idx in text.data[0]:
        #         print(token_idx.item(), text_field.vocab.itos[token_idx.item()])
        #
        #     break

    @staticmethod
    def _tokenize_text(text):
        return text.strip().split()
