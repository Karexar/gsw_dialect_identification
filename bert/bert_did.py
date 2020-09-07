import torch
import argparse
from tqdm import tqdm
from transformers import BertForSequenceClassification as BertForSeqClass
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score
from loss_functions.focalloss import FocalLoss
from loss_functions.smooth_labels import LabelSmoothingCrossEntropy
import numpy as np
import pandas as pd
import random
import os
import time
import datetime
from typechecker.typecheck import *
from bert.data_manager import DataManager
from bert.tokenizer import Tokenizer
from bert.bert import Bert
from pathlib import Path
#import _thread
from utils.utils import *


class BertDid:
    def __init__(self,
                 model_dir,
                 data_dir=None,
                 fine_tune_model=False,
                 pretrained_model_name="bert-base-german-cased",
                 sentence_length=128,
                 lower_case=False,
                 batch_size=32,
                 cuda_device=0,
                 label_names=None,
                 soft_labels=False,
                 weight_soft=1.0,
                 focal_loss_gamma=0.0,
                 focal_loss_alpha=None):

        # Set the seed value all over the place for reproducibility
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Prepare the GPU
        self.device = None
        if torch.cuda.is_available() and cuda_device is not None:
            self.device = torch.device("cuda")
            torch.cuda.set_device(cuda_device)
            print(f'Using GPU: {torch.cuda.current_device()}')
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        # Create the BERT model
        self.classifier = Bert(focal_loss_gamma,
                               focal_loss_alpha,
                               sentence_length)

        # Instantiate the tokenizer
        self.tokenizer = Tokenizer(pretrained_model_name, lower_case)

        if not fine_tune_model:
            # Load the fine-tuned BERT model and tokenizer
            self.load(model_dir)
            self.classifier.model = self.classifier.model.to(self.device)

        # Load the dataset for training and/or evaluating. This is not needed if
        # the model is only used to predict new sentences

        self.data_manager = DataManager(data_dir,
                                        self.tokenizer,
                                        batch_size,
                                        label_names=label_names,
                                        soft_labels=soft_labels,
                                        weight_soft=weight_soft)
        self.soft_labels = soft_labels

    def load(self, model_dir):
        """Load an existing model and tokenizer

        Parameters
            model_dir | str
                The directory path containing the fine-tuned BERT model
        """
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            # Load a trained model and vocabulary that you have fine-tuned
            self.classifier.load_finetuned(model_dir)
            self.tokenizer.load(model_dir)
        elif os.path.exists(model_dir):
            raise ValueError("No model in the specified directory")
        else:
            raise ValueError("The given directory does not exist")


    def save(self, model_dir):
        """Save the current model and tokenizer

        Parameters
            model_dir | str
                The directory where to save the model
        """
        # Create output directory if needed
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        print(f"Saving model to {model_dir}")
        self.classifier.save(model_dir)
        self.tokenizer.save(model_dir)


    def flat_accuracy(self, preds, labels):
        """ Calculate the accuracy of our predictions vs labels

        Parameters
            preds | list[list[float]]
                The list of logits
            labels | list[list[int]]
                The list of label indices
        """
        if self.soft_labels:
            labels =  np.argmax(labels, axis=1).flatten()
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        """ Convert a time in seconds into hh:mm:ss

        Parameters
            elapsed | int
                The elpased time in seconds
        """
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    def fine_tune(self,
                  learning_rate,
                  epochs,
                  pretrained_model_name,
                  labels_count):
        # Load BertForSequenceClassification, the pretrained BERT model with a
        # single linear classification layer on top.

        self.classifier.load_pretrained(pretrained_model_name, labels_count)
        self.classifier.model = self.classifier.model.to(self.device)

        # Tell pytorch to run this model on the GPU.
        #self.classifier.model.cuda()

        # Get all of the model's parameters as a list of tuples.
        params = list(self.classifier.model.named_parameters())

        print(f'The BERT model has {len(params)} different named parameters.\n')

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Optimizer and learning rate scheduler

        # For the purposes of fine-tuning, the authors recommend choosing from
        # the following values :
        # Batch size: 16, 32
        # Learning rate (Adam): 5e-5, 3e-5, 2e-5
        # Number of epochs: 2, 3, 4

        optimizer = AdamW(self.classifier.model.parameters(),
                          lr = learning_rate, # default is 5e-5
                          eps = 1e-8 # default is 1e-8.
                        )

        # Total number of training steps is [number of batches] x [number of
        # epochs].
        total_steps = len(self.data_manager.train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                            # 0 = Default value in run_glue.py
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/
        #     5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # We'll store a number of quantities such as training and validation
        # loss, validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode.
            # `dropout` and `batchnorm` layers behave differently during
            # training vs. test (source: https://stackoverflow.com/questions/
            #    51433378/what-does-model-train-do-in-pytorch)
            self.classifier.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.data_manager.train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                                    step, len(self.data_manager.train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU
                # using the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before
                # performing a backward pass. PyTorch doesn't do this
                # automatically because accumulating the gradients is
                # "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/
                #   why-do-we-need-to-call-zero-grad-in-pytorch)
                self.classifier.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training
                # batch). The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/
                #    bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what
                # arguments arge given and what flags are set. For our useage
                # here, it returns the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.

                logits = self.classifier.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)
                logits = logits[0]
                loss_fct = FocalLoss(gamma=self.classifier.focal_loss_gamma,
                                     alpha=self.classifier.focal_loss_alpha,
                                     soft_labels=self.soft_labels)
                labels = b_labels
                if not self.soft_labels:
                    labels = b_labels.view(-1)
                loss = loss_fct(logits.view(-1, self.data_manager.labels_count),
                                labels)



                # Accumulate the training loss over all of the batches so that
                # we can calculate the average loss at the end. `loss` is a
                # Tensor containing a single value; the `.item()` function just
                # returns the Python value from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.classifier.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters
                # are modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.data_manager.train_dataloader)

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our
            # performance on our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            self.classifier.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.data_manager.validation_dataloader:

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU
                # using the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph
                # during the forward pass, since this is only needed for
                # backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/
                    #    bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the
                    # output values prior to applying an activation function
                    # like the softmax.
                    logits = self.classifier.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask)
                    logits = logits[0]
                    loss_fct = FocalLoss(gamma=self.classifier.focal_loss_gamma,
                                         alpha=self.classifier.focal_loss_alpha,
                                         soft_labels=self.soft_labels)
                    labels = b_labels
                    if not self.soft_labels:
                        labels = b_labels.view(-1)
                    loss = loss_fct(logits.view(-1, self.data_manager.labels_count),
                                    labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)


            # Report the final accuracy for this validation run.
            avg_val_accuracy = (total_eval_accuracy /
                                len(self.data_manager.validation_dataloader))
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.data_manager.validation_dataloader)

            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(
                                        self.format_time(time.time()-total_t0)))

        pd.set_option('precision', 2) # Display floats with two decimal places.
        df_stats = pd.DataFrame(data=training_stats)
        df_stats = df_stats.set_index('epoch')
        print(df_stats)


    def evaluate(self):
        """Make prediction on the test set
        """
        # Put model in evaluation mode
        self.classifier.model.eval()

        # Tracking variables
        predictions , true_labels = [], []
        # Predict
        for batch in self.data_manager.test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory
            # and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.classifier.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Compute the true values and prediction list
        true_list = []
        pred_list = []
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                pred = self.data_manager.label_int_to_str[np.argmax(predictions[i][j])]
                pred_list.append(pred)
                if self.soft_labels:
                    true_value = self.data_manager.label_int_to_str[np.argmax(true_labels[i][j])]
                else:
                    true_value = self.data_manager.label_int_to_str[true_labels[i][j]]
                true_list.append(true_value)

        # Print the scores
        accuracy = accuracy_score(true_list, pred_list)
        f1 = dict()
        for average in ["micro", "macro", "weighted"]:
            f1[average] = f1_score(true_list,
                                   pred_list,
                                   self.data_manager.labels_str,
                                   average=average)

        print("F1_micro : " + str(round(f1["micro"], 3)))
        print("F1_macro : " + str(round(f1["macro"], 3)))
        print("F1_weighted : " + str(round(f1["weighted"], 3)))
        print("Accuracy : " + str(round(accuracy, 3)))
        print("Recall : ")
        recall = recall_score(true_list,
                              pred_list,
                              self.data_manager.labels_str,
                              average=None)
        recall = [round(x, 3) for x in recall]
        for i in range(len(recall)):
            print(f"{self.data_manager.labels_str[i]} : {recall[i]}")

        print("Precision : ")
        precision = precision_score(true_list,
                                    pred_list,
                                    self.data_manager.labels_str,
                                    average=None)
        precision = [round(x, 3) for x in precision]
        for i in range(len(recall)):
            print(f"{self.data_manager.labels_str[i]} : {precision[i]}")

        label_names = self.data_manager.labels_str
        print("Percentage of prediction for each class :")
        counts = pd.Series(pred_list).value_counts().sort_index()
        for i in range(len(counts)):
            print(f"{label_names[i]} : {round(counts[i] / counts.sum(), 3)}")
        print("---")
        print(recall)
        for x in recall:
            print(x)
        print(precision)
        for x in precision:
            print(x)
        print(self.data_manager.labels_str)
        print("---")

    def predict(self, sentences):
        """Make prediction on a given list of sentences

        Parameters
            sentences | str
                A list of sentences to predict or a path to a csv file
                containing sentences to predict in the first column.
        Returns
            Tuple[list[str], list[list[float]]]
                A tuple of two elements. The first one represents the labels
                values, and the second one is a list containing the dialect
                probabibilities for each sentence.
        """
        dataloader=None
        if isinstance(sentences, str):
            df = pd.read_csv(sentences, header=None, sep="\t")
            sentences = list(df.iloc[:, 0].values)
        self.data_manager.prepare_predict_dataloader(
                                            sentences,
                                            self.classifier.sentence_length)

        print('Predicting labels...')

        # Put model in evaluation mode
        self.classifier.model.eval()

        # Tracking variables
        predictions = []

        # Predict
        for batch in tqdm(self.data_manager.prediction_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory
            # and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.classifier.model(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions
            predictions.append(logits)

        probabilities = [np.exp(x)/sum(np.exp(x))
                            for y in predictions for x in y]
        return (self.data_manager.labels_str, probabilities)


if __name__ == '__main__':
    #_thread.start_new_thread( keep_alive, tuple() )

    parser = argparse.ArgumentParser(
                            description='Main entry point for BERT model',
                            add_help=True)

    group = parser.add_argument_group('BERT')
    group.add_argument('--pretrained_model_name',
                       type=str,
                       default="bert-base-german-cased",
                       help="Name of the pretrained BERT model to use")
    group.add_argument('--data_dir',
                       type=str,
                       default=None,
                       help="Directory containing the input dataset")
    group.add_argument('--model_dir',
                       type=str,
                       help="Directory where to load/save the finetuned model")
    group.add_argument('--cuda_device',
                       default=None,
                       type=int,
                       help="Cuda device index to use")
    group.add_argument('--sentence_length',
                       type=int,
                       help="Sentence length (count of tokens) used by BERT")
    # Batch size for training. For fine-tuning BERT on a specific task, the
    # authors recommend a batch size of 16 or 32.
    group.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help="Batch size for training (16 or 32 recommended)")
    group.add_argument('--epochs',
                       type=int,
                       default=3,
                       help="Epochs for training (2, 3, or 4 recommended)")
    group.add_argument('--learning_rate',
                       type=float,
                       default=2e-5,
                       help="Learning rate for fine-tuning the model " +
                            "(5e-5, 3e-5, or 2e-5 recommended)")
    group.add_argument('--lower_case',
                       dest='lower_case',
                       action='store_true',
                       help="Whether or not to lower case the input dataset")
    group.add_argument('--fine_tune_model',
                       dest='fine_tune_model',
                       action='store_true',
                       help="Whether or not to fine_tune the model")
    group.add_argument('--evaluate_model',
                       dest='evaluate_model',
                       action='store_true',
                       help="Whether or not to evaluate the model")
    group.add_argument('--path_sentences',
                       type=str,
                       default=None,
                       help="Path to sentences to predict dialect" )
    group.add_argument('--compute_max_length',
                       dest='compute_max_length',
                       action='store_true',
                       help="Whether or not to compute the encoded max " +
                            "sentence length")
    group.add_argument('--soft_labels',
                       dest='soft_labels',
                       action='store_true',
                       help="Whether or not to use soft labels to train the " +
                            "model")
    group.add_argument('--weight_soft',
                       type=float,
                       default=0.0,
                       help="Weight given to the soft label predicted for " +
                       "interpolation with the hard label predicted")
    group.add_argument('--focal_loss_gamma',
                       type=float,
                       default=0.0,
                       help="The gamma value of focal loss. If 0.0, then " +
                            "this is equivalent to cross-entropy")
    group.add_argument('--focal_loss_alpha',
                       nargs="+",
                       type=float,
                       default=None,
                       help="The alpha value of focal loss")
    group.add_argument('--label_names',
                       type=str,
                       default=None,
                       help="Comma-separated string representing the sorted " +
                       "label names to use. Default will infer the labels " +
                       "from the data, which may be a problem if we are " +
                       "evaluating on a dataset which does not contain all " +
                       "the labels used for training")

    cli_args = parser.parse_args()

    if cli_args.soft_labels and cli_args.label_names is None:
        raise ValueError("label_names must be provided when soft labels are " +
                         "used")

    if cli_args.fine_tune_model:
        Path(cli_args.model_dir).mkdir(parents=True, exist_ok=True)

    # Load an already fine-tuned model if we are not fine-tuning a
    # pretrained BERT model
    if not cli_args.fine_tune_model:
        # Overwrite the parameters to match the training parameters
        path = os.path.join(cli_args.model_dir, "training_args.bin")
        args = torch.load(path)
        cli_args.batch_size = args.batch_size
        cli_args.sentence_length = args.sentence_length
        cli_args.pretrained_model_name = args.pretrained_model_name
        cli_args.lower_case = args.lower_case
        cli_args.label_names = args.label_names

    label_names = cli_args.label_names
    if isinstance(label_names, str):
        label_names = label_names.split(',')

    print("Initialising...")
    bert_did = BertDid(cli_args.model_dir,
                       cli_args.data_dir,
                       cli_args.fine_tune_model,
                       cli_args.pretrained_model_name,
                       cli_args.sentence_length,
                       cli_args.lower_case,
                       cli_args.batch_size,
                       cli_args.cuda_device,
                       label_names,
                       cli_args.soft_labels,
                       cli_args.weight_soft,
                       cli_args.focal_loss_gamma,
                       cli_args.focal_loss_alpha)

    #cli_args.focal_loss_gamma if hasattr(cli_args, "focal_loss_gamma") else 0.0,
    #cli_args.focal_loss_alpha if hasattr(cli_args, "focal_loss_alpha") else None)

    if cli_args.compute_max_length:
        print("Computing max sentence length...")
        sentences = bert_did.data_manager.get_all_sentences()
        max_length = bert_did.tokenizer.get_max_sentence_length(sentences)
        if cli_args.sentence_length < max_length:
            print("WARNING : specified sentence length is smaller than max " +
                  "sentence length in dataset")

    if cli_args.fine_tune_model:
        print("Preparing the training and evaluation set...")
        bert_did.data_manager.prepare_train_eval_dataloader(cli_args.sentence_length)
        print("Fine-tuning the model...")
        bert_did.fine_tune(cli_args.learning_rate,
                           cli_args.epochs,
                           cli_args.pretrained_model_name,
                           bert_did.data_manager.labels_count)

    if cli_args.evaluate_model:
        print("Preparing the test set...")
        bert_did.data_manager.prepare_test_dataloader(cli_args.sentence_length)
        print("Evaluate on test set...")
        bert_did.evaluate()

    if cli_args.path_sentences is not None:
        print("Predicting the dialect...")
        preds = bert_did.predict(cli_args.path_sentences)
        print(preds[0])
        for x in preds[1][0]:
            print(x)

    if cli_args.fine_tune_model:
        print("Saving current model on disk...")
        bert_did.save(cli_args.model_dir)
        path = os.path.join(cli_args.model_dir, "training_args.bin")
        cli_args.__dict__.update(label_names=bert_did.data_manager.labels_str)
        torch.save(cli_args, path)
