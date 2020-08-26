"""
(c) Copyright 2019 Swisscom AG
All Rights Reserved.
"""
import argparse


class Arguments:

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            parser = argparse.ArgumentParser(description='Train and use an LSTM language model')

        group = parser.add_argument_group('LSTM')
        group.add_argument('--data', type=str, default='./data/', help='Location of the data corpus')
        group.add_argument('--num_hidden', type=int, default=200, help='Number of hidden units per layer')
        group.add_argument('--num_layers', type=int, default=2, help='Number of layers')
        group.add_argument('--learning_rate', type=float, default=20, help='Initial learning rate')
        group.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
        group.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
        group.add_argument('--batch_size', type=int, default=20, metavar='N', help='Batch size')
        group.add_argument('--eval_batch_size', type=int, default=10, metavar='N', help='Evaluation batch size')
        group.add_argument('--bptt', type=int, default=35, help='Sequence length')
        group.add_argument('--dropout', type=float, default=0.2,
                           help='Dropout applied to layers (0 = no dropout)')
        group.add_argument('--tied', action='store_true', help='Tie the word embedding and softmax weights')
        group.add_argument('--seed', type=int, default=1111, help='Random seed')
        group.add_argument('--cuda', action='store_true', help='Use CUDA')
        group.add_argument('--cuda-device', dest='cuda_device', default='cuda:0', help='Which CUDA-enabled GPU to use')
        group.add_argument('--log_interval', type=int, default=200, metavar='N', help='Report interval')
        group.add_argument('--model-path-lstm', dest='model_path_lstm', type=str, default='model.pt',
                           help='Path to save the final model')
        group.add_argument('--train-lstm', dest='train_lstm', action='store_true',
                           help='Indicates whether a new LSTM model should be trained')
        group.add_argument('--eval-lstm', dest='eval_lstm', action='store_true',
                           help='Indicates that the given LSTM model will be evaluated on the test set')
        group.add_argument('--freeze-embed', dest='freeze_embed', action='store_true',
                           help='Indicates whether the pre-trained word embeddings should be further trained '
                                'on the given training data or frozen')
        group.add_argument('--use-pretrained-embed', dest='use_pretrained_embed', action='store_true',
                           help='Indicates whether a to use a pre-trained word embeddings model')
        group.add_argument('--patience', dest='patience', type=int, default=2,
                           help='Number of epochs to wait for improvement on the validation perplexity before '
                                'early stopping the training process')
        group.add_argument('--stop-thresh', dest='stop_thresh', type=float, default=0.1,
                           help='The improvement that the validation perplexity score needs to exhibit '
                                'after every epoch to avoid early stopping')

        return parser
