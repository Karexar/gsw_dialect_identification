"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import os
import time

import torch

from lstm.models.lstm import LSTM
from sklearn.metrics import f1_score


class ModelExecutor:

    def __init__(self, model, data_manager, device, criterion=None):
        self.model = model
        self.data_manager = data_manager
        self.device = device
        self.criterion = criterion

    def train(self, epoch, batch_size, learning_rate, model_optimiser, clip, log_interval):
        total_loss = 0
        total_accuracy = 0

        # Turn on training mode which enables dropout.
        self.model.train()
        start_time = time.time()

        num_labels = self.data_manager.num_labels
        hidden = self.model.init_hidden(batch_size)

        for idx, batch in enumerate(self.data_manager.train_iter):
            text = batch.text[0].to(self.device)
            targets = torch.autograd.Variable(batch.dialect).long().to(self.device)
            # One of the batch returned by BucketIterator has length different than batch_size
            if text.size()[0] is not batch_size:
                continue

            #print()
            #print(f'Idx: {idx} / {len(self.data_manager.train_iter)} batches')
            #print(f'Input text: {text.size()}')
            #print(f'Targets: {targets.size()}')
            #print()

            #print(f'Original targets: {targets}')
            #print(f'Target modification: {torch.autograd.Variable(targets).long()}')
            #print()

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = LSTM.repackage_hidden(hidden)
            # hidden = self.model.init_hidden(batch_size)
            model_optimiser.zero_grad()

            #print("---")
            #print(type(self.model))
            #print(type(text))
            #print(text)
            #print(type(hidden))
            #print(hidden)
            #print("---")
            output, hidden, _ = self.model(text, hidden)

            #print(f'Output: {output.size()}')
            #print(f'Hidden state: {hidden[0].size()}')
            #print(f'Predictions: {output.view(-1, num_labels).size()}')
            #print(f'Targets: {targets}')
            #print()

            loss = self.criterion(output.view(-1, num_labels), targets)
            num_corrects = (torch.max(output, 1)[1].data == targets.data).float().sum()
            accuracy = 100.0 * num_corrects / len(batch)

            #print()
            #print(f'Correct predictions: {torch.max(output, 1)[1].data == targets.data}')
            #print(f'Loss: {loss}')
            #print(f'Accuracy: {accuracy}')
            #print()

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            model_optimiser.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()

            #print(f'Total loss: {total_loss}')
            #print(f'Total accuracy: {total_accuracy}')

            if idx % log_interval == 0 and idx > 0:
                current_loss = total_loss / (idx + 1)
                current_accuracy = total_accuracy / (idx + 1)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:6.2f} | '
                      'loss {:5.2f} | accuracy {:8.2f}'.format(
                    epoch, idx, len(self.data_manager.train_iter), learning_rate,
                    elapsed * 1000 / log_interval, current_loss, current_accuracy))
                start_time = time.time()

        return total_loss / len(self.data_manager.train_iter), total_accuracy / len(self.data_manager.train_iter)

    def evaluate(self, data_source, batch_size):
        total_loss = 0
        total_accuracy = 0

        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        num_labels = self.data_manager.num_labels
        # hidden = self.model.init_hidden(batch_size)
        target_values = []
        pred_values = []
        with torch.no_grad():
            for idx, batch in enumerate(data_source):
                text = batch.text[0].to(self.device)
                targets = torch.autograd.Variable(batch.dialect).long().to(self.device)
                # One of the batch returned by BucketIterator has length different than batch_size
                if text.size()[0] is not batch_size:
                    continue

                # hidden = LSTM.repackage_hidden(hidden)
                hidden = self.model.init_hidden(batch_size)
                output, hidden, _ = self.model(text, hidden)

                predictions = torch.max(output, 1)[1]
                loss = self.criterion(output.view(-1, num_labels), targets)
                # print(f'Predictions: {torch.max(output, 1)[1].data}')
                # print(f'Targets: {targets.data}')
                num_corrects = (predictions.data == targets.data).float().sum()
                accuracy = 100.0 * num_corrects / len(batch)

                total_loss += loss.item()
                total_accuracy += accuracy.item()

                target_values.extend(targets.data.cpu())
                pred_values.extend(predictions.data.cpu())
        f1_micro = f1_score(target_values, pred_values, average="micro")
        f1_macro = f1_score(target_values, pred_values, average="macro")
        print(f"F1-micro : {f1_micro}")
        print(f"F1-macro : {f1_macro}")

        return total_loss / len(data_source), total_accuracy / len(data_source)

    def predict_dialect_label(self, input_tokens):
        self.model.eval()

        hidden = self.model.init_hidden(1)
        with torch.no_grad():
            output, hidden, sentence_embedding = self.model(input_tokens, hidden)
            prediction_idx = torch.max(output, 1)[1].item()
            prediction = self.data_manager.idx2dialect[prediction_idx]

        return prediction, sentence_embedding.flatten().tolist()

    def load_pre_trained_model(self, model_path, device):
        assert os.path.exists(model_path)
        self.model = LSTM.import_model(model_path, device)
