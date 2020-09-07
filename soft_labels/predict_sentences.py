# This scripts predicts sentences to form a dataset of soft labelled sentences
# that will be used as a training set for the final model


from bert.bert_did import *
import pandas as pd
import os
from pathlib import Path

###  Settings  ################################################################
dataset_path = "data/twitter/gsw_filtered.csv"
labelled_test_path = "data/twitter/labelled/test.csv"
soft_labels_dir = "data/twitter/soft_labels/"
model_dir = "models/bert/twitter_german_cased_mix_e3_g2_s128"
pretrained_model_name = "bert-base-german-cased"
sentence_length = 128
cuda_device = 0
test_size = 0.1
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
weight_prediction = 0 # weight to give to soft vs hard label predicted
sample_frac=0.01
###############################################################################

output_dir = os.path.join(soft_labels_dir, "frac_" + str(sample_frac))
output_dir = os.path.join(output_dir, "weight_" + str(weight_prediction))
Path(output_dir).mkdir(parents=True, exist_ok=True)

print("Loading the dataset...")
df = pd.read_csv(dataset_path, sep="\t")
print("Filtering...")
# keep only sentences with very high probability of being Swiss-German
df = df[df.prediction >= 0.99]
# remove sentences that appear in the test set of the labelled dataset
df_lab_test = pd.read_csv(labelled_test_path, sep="\t", header=None)
test_sentences = set(df_lab_test.iloc[:, 0].values)
df = df[~df.sentence.isin(test_sentences)]
df = pd.DataFrame(df.sentence)
df = df.sample(frac=sample_frac)
print(f"Length dataset : {df.shape[0]}")

# predict
print("Initialising dialect identification model...")
bert_did = BertDid(model_dir,
                   pretrained_model_name=pretrained_model_name,
                   sentence_length=sentence_length,
                   cuda_device=cuda_device,
                   label_names=label_names)

sentences = list(df.sentence.values)

_, logits = bert_did.predict(sentences)

hard_predictions = np.array([[0 if i != np.argmax(x) else 1
                              for i in range(len(x))] for x in logits])

def linear_interpolation(x, y, e):
    return e*x + (1.0-e)*y

print(logits[0])
if weight_prediction != 1:
    logits = [linear_interpolation(logits[i],
                                   hard_predictions[i] ,
                                   weight_prediction)
              for i in range(len(logits))]
print(logits[0])
logits = [[str(y) for y in x] for x in logits]
logits = [','.join(x) for x in logits]


df["predictions"] = logits
idx = int(test_size*df.shape[0])
df_test = df.iloc[:idx, :]
df_train = df.iloc[idx:, :]
print(df_test.head())
print(df_train.head())

print("Saving the train and test set")
path = os.path.join(output_dir, "test.csv")
df_test.to_csv(path, sep="\t", index=False, header=None)
path = os.path.join(output_dir, "train.csv")
df_train.to_csv(path, sep="\t", index=False, header=None)
