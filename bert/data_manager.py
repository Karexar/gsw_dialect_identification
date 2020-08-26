import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import os


class DataManager:
    def __init__(self,
                 data_dir,
                 tokenizer,
                 batch_size,
                 sep="\t",
                 dev_ratio=0.1,
                 label_names=None):
        """Load the dataset found in the given folder. It expects 'train.csv',
        'test.csv', and optionnaly 'dev.csv' in which case the file will be
        merged with train.csv. The first column is the text and the second
        column is the label.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.labels_str = label_names
        # Train set
        if not data_dir is None:
            df_train = pd.read_csv(os.path.join(data_dir, "train.csv"),
                                        sep=sep,
                                        names=["Text", "Label"])
            print(f"Train set size : {df_train.shape[0]}")
            # Concat train and dev if available
            if "dev.csv" in os.listdir(data_dir):
                print("dev.csv file found, merging with train.csv...")
                df_dev = pd.read_csv(os.path.join(data_dir, "dev.csv"),
                                     sep=sep,
                                     names=["Text", "Label"])
                print(f"Dev set size : {df_dev.shape[0]}")
                df_train = pd.concat([df_train, df_dev], axis=0)
                print(f"Train+Dev set size : {df_train.shape[0]}")
            # Get the lists of sentences and their labels.
            self.train_sentences = df_train.Text.values
            train_labels_str = list(df_train.Label.values)

            # Test set
            df_test = pd.read_csv(os.path.join(data_dir, "test.csv"),
                                   sep=sep,
                                   names=["Text", "Label"])
            print(f"Test set size : {df_test.shape[0]}")
            # Create sentence and label lists
            self.test_sentences = df_test.Text.values
            test_labels_str = list(df_test.Label.values)

            labels_str_from_data = sorted(list(set(train_labels_str +
                                                   test_labels_str)))
            if label_names is None:
                self.labels_str = labels_str_from_data
            else:
                # Check if the given labels names match the labels names
                # inferred from the data, which may not be the case if the
                # evaluation dataset does not contains all labels found during
                # finetuning on the training set.
                if len(self.labels_str) != len(labels_str_from_data):
                    print("WARNING : given label names count " +
                          f"({len(self.labels_str)}) does not match inferred " +
                          f"label count ({len(labels_str_from_data)})")

        self.labels_count = len(self.labels_str)
        print(f'Number of labels: {self.labels_count}')

        label_str_to_int = dict()
        self.label_int_to_str = dict()
        for i in range(self.labels_count):
            label_str_to_int[self.labels_str[i]] = i
            self.label_int_to_str[i] = self.labels_str[i]

        if not data_dir is None:
            self.train_labels = [label_str_to_int[x] for x in train_labels_str]
            self.test_labels = [label_str_to_int[x] for x in test_labels_str]


    def prepare_train_eval_dataloader(self,
                                      sentence_length,
                                      dev_ratio=0.1):
        # Tokenize the training sentences
        input_ids, attention_masks = self.tokenizer.tokenize(self.train_sentences,
                                                             sentence_length)
        labels = torch.tensor(self.train_labels)

        # Create a 90-10 train-validation split.
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Calculate the number of samples to include in each set.
        train_size = int((1.0-dev_ratio) * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset,
                                                  [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        self.train_dataloader = DataLoader(
                                    train_dataset,
                                    sampler = RandomSampler(train_dataset),
                                    batch_size = self.batch_size
                                )

        # For validation the order doesn't matter, so we'll just read them
        # sequentially.
        self.validation_dataloader = DataLoader(
                                    val_dataset,
                                    sampler = SequentialSampler(val_dataset),
                                    batch_size = self.batch_size
                                )

    def prepare_test_dataloader(self, sentence_length):
        # Tokenize the test sentences
        input_ids, attention_masks = self.tokenizer.tokenize(self.test_sentences,
                                                             sentence_length)

        labels = torch.tensor(self.test_labels)

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        self.test_dataloader = DataLoader(prediction_data,
                                           sampler=prediction_sampler,
                                           batch_size=self.batch_size)

    def prepare_predict_dataloader(self, sentences, sentence_length):
        # Tokenize the test sentences
        input_ids, attention_masks = self.tokenizer.tokenize(sentences,
                                                             sentence_length)

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        self.prediction_dataloader = DataLoader(prediction_data,
                                             sampler=prediction_sampler,
                                             batch_size=self.batch_size)

    def get_all_sentences(self):
        return list(self.train_sentences) + list(self.test_sentences)
