# This script predict all sentences of a dataset.
# It produces a csv file with first column being the sentence, second column
# being the user id, and the remaining columns being the probability for each
# dialect

from bert.bert_did import *
import pandas as pd
import os
#import _thread
from utils.utils import *
import math

###  Settings  ################################################################
input_path = "data/twitter/unlabelled/full.csv"
output_path = "data/twitter/unlabelled/full_predicted.csv"
model_dir = "models/bert/twitter_german_cased_e3_g2_s128"
pretrained_model_name = "bert-base-german-cased"
sentence_length = 128
cuda_device = 0
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
columns = ["sentence", "user_id"]
###############################################################################

#_thread.start_new_thread( keep_alive, tuple() )

print("Initialising...")
bert_did = BertDid(model_dir,
                   pretrained_model_name=pretrained_model_name,
                   sentence_length=sentence_length,
                   cuda_device=cuda_device,
                   label_names=label_names)

df = pd.read_csv(input_path, header=None, sep="\t", names=columns)

sentences = list(df.sentence.values)

predictions = bert_did.predict(sentences)

for i in range(len(label_names)):
    df[label_names[i]] = [x[i] for x in predictions[1]]

df.to_csv(output_path, sep="\t", index=False)
