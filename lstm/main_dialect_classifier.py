import argparse
import random
import time

import neptune
import torch
from torch import nn, optim

from lstm.arguments.cli_arguments import Arguments
from lstm.data_manager.data_manager import DataManager
from lstm.models.lstm import LSTM
from lstm.models.model_executor import ModelExecutor
from word_representations.word_embeddings.glove import GloveModel
from word_representations.word_embeddings.word2vec import Word2VecModel
from word_representations.word_embeddings.word_embeddings_model import WordEmbeddingsModel


def training_pipeline(args):
    ###############################################################################
    # Environment setup
    ###############################################################################

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Check if CUDA device is available and set training on CPUs or GPUs
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device(args.cuda_device if args.cuda else "cpu")

    ###############################################################################
    # Experiment tracking setup
    ###############################################################################
    neptune.init(project_qualified_name='karexar/GSW-dialect-classifier')
    args_dict = vars(args)
    neptune.create_experiment(params=args_dict)
    if hasattr(args, 'experiment_id'):
        neptune.append_tag(args.experiment_id)
    neptune.set_property('lm_algo', 'lstm')
    for key in args_dict.keys():
        neptune.set_property(key, args_dict[key])

    ###############################################################################
    # Load data
    ###############################################################################

    print('Loading data')
    data_manager = DataManager(args.data, device, args.batch_size, args.eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################

    print('Building model')
    num_tokens = data_manager.vocab_size
    num_labels = data_manager.num_labels
    embeddings_matrix = None
    if args.use_pretrained_embed:
        # Load pre-trained word embeddings model
        # and generate the embeddings weight matrix for the entire vocabulary
        assert args.embed_algo is not None
        print(f'Using {args.embed_algo} pre-trained word embeddings')

        if args.embed_algo == 'word2vec':
            pretrained_embeddings = Word2VecModel(args.model_path_embed,
                                                  args.model_name_embed,
                                                  load_from_disk=True)
            embeddings_matrix = pretrained_embeddings.get_vocabulary_embeddings(
                data_manager.idx2word, args.embed_size
            )
        elif args.embed_algo == 'glove':
            pretrained_embeddings = GloveModel(args.model_path_embed,
                                               args.model_name_embed,
                                               load_from_disk=True)
            embeddings_matrix = pretrained_embeddings.get_vocabulary_embeddings(
                data_manager.idx2word, args.embed_size
            )

    model = LSTM(
        num_tokens,
        args.embed_size,
        args.num_hidden,
        args.num_layers,
        args.dropout,
        num_labels,
        embeddings_matrix
    ).to(device)

    print('Model architecture')
    print(model)

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    print('Initialising model executor')
    model_executor = ModelExecutor(model, data_manager, device, criterion)

    if args.train_lstm:
        # Loop over epochs
        learning_rate = args.learning_rate
        best_val_accuracy = None
        last_val_accuracy = 0
        model_optimiser = optim.SGD(model.parameters(), lr=learning_rate)

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('Starting the training process')
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()

                _, _ = model_executor.train(epoch, args.batch_size, learning_rate,
                                            model_optimiser, args.clip, args.log_interval)
                val_loss, val_accuracy = model_executor.evaluate(data_manager.val_iter, args.eval_batch_size)

                # Log result in Neptune ML
                neptune.send_metric('valid_loss', epoch, val_loss)
                neptune.send_metric('valid_accuracy', epoch, val_accuracy)
                neptune.send_metric('learning_rate', epoch, learning_rate)

                if epoch % 3 == 0:
                    learning_rate *= 0.9  # correct the learning rate after some number of epochs

                print('-' * 89)
                print('| End of epoch {:3d} | Time: {:5.2f}s | Valid loss {:6.2f} | '
                      'Valid accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                      val_loss, val_accuracy))
                print('-' * 89)

                # Save the model if the validation accuracy is the best we've seen so far.
                if not best_val_accuracy or val_accuracy > best_val_accuracy:
                    model_executor.model.export_model(args.model_path_lstm)
                    best_val_accuracy = val_accuracy

                if val_accuracy < last_val_accuracy:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    learning_rate /= 2.0

                for group in model_optimiser.param_groups:
                    group['lr'] = learning_rate

                last_val_accuracy = val_accuracy

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    ###############################################################################
    # Evaluation code
    ###############################################################################

    test_loss = None
    test_accuracy = None
    if args.eval_lstm:
        print('Evaluating on the test set')

        # Load the best saved model.
        model_executor.load_pre_trained_model(args.model_path_lstm, device=device)

        # Run on test data.
        test_loss, test_accuracy = model_executor.evaluate(data_manager.test_iter,
                                                           args.eval_batch_size)

        # Log result in Neptune ML
        neptune.send_metric('test_loss', test_loss)
        neptune.send_metric('test_accuracy', test_accuracy)

        print('-' * 89)
        print('| End of evaluation | Test loss {:6.2f}'.format(test_loss)
              + ' | Test accuracy {:8.2f}'.format(test_accuracy))
        print('-' * 89)

    ###############################################################################
    # Stop the experiment tracking
    ###############################################################################

    neptune.stop()

    return test_loss, test_accuracy


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Main entry point for LSTM model', add_help=True)
    subparsers = main_parser.add_subparsers(title='Word embeddings type', dest='embed_algo',
                                            description='The word embeddings algorithm to use')

    Arguments.fetch_cli_arguments(main_parser)
    WordEmbeddingsModel.fetch_cli_arguments(main_parser)

    _ = subparsers.add_parser('random', help='Use a random initialisation for the weights of the embeddings layer')

    w2v_parser = subparsers.add_parser('word2vec', help='Use a Word2Vec word embeddings model')
    Word2VecModel.fetch_cli_arguments(w2v_parser)

    glove_parser = subparsers.add_parser('glove', help='Use a GloVe word embeddings model')
    GloveModel.fetch_cli_arguments(glove_parser)

    cli_args = main_parser.parse_args()
    training_pipeline(cli_args)
