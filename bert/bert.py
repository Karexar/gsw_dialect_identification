from transformers import BertForSequenceClassification as BertForSeqClass
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from loss_functions.focalloss import FocalLoss
import pandas as pd
import numpy as np
import time
import datetime

class Bert:
    def __init__(self,
                 focal_loss_gamma,
                 focal_loss_alpha,
                 sentence_length):
        self.model = None
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.sentence_length = sentence_length

    def load_finetuned(self, model_dir):
        """Load an already fine-tuned model from disk.
        """
        self.model = BertForSeqClass.from_pretrained(model_dir)

    def load_pretrained(self, pretrained_model_name, labels_count):
        self.model = BertForSeqClass.from_pretrained(
            pretrained_model_name,
            num_labels = labels_count,
            output_attentions = False,
            output_hidden_states = False
        )

    def save(self, model_dir):
        """Save the model on disk
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') \
                                          else self.model
        model_to_save.save_pretrained(model_dir)
