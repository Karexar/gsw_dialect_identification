# This script predicts the dialect for each sentence of a dataset

from bert.bert_did import *
import pandas as pd

###  Settings  ################################################################
dataset_path = "data/whatsapp/labelled/full.csv"
model_dir = "models/bert/twitter_german_cased_mix_e3_g2_s128"
pretrained_model_name = "bert-base-german-cased"
sentence_length = 128
cuda_device = 0
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
output_path = "data/whatsapp/labelled/predicted.csv"
###############################################################################


print("Loading the dataset...")
df = pd.read_csv(dataset_path, sep="\t")

# Filter out short sentences
lengths = df.iloc[:, 0].apply(lambda x: len(x.split()))
df = df[lengths >= 5]

# predict
print("Initialising dialect identification model...")
bert_did = BertDid(model_dir,
                   pretrained_model_name=pretrained_model_name,
                   sentence_length=sentence_length,
                   cuda_device=cuda_device,
                   label_names=label_names)

sentences = list(df.iloc[:, 0].values)

print("Predicting dialects...")
_, logits = bert_did.predict(sentences)

df["dialect_predicted"] = [label_names[np.argmax(x)] for x in logits]
df["logits"] = [','.join([str(y) for y in x]) for x in logits]

print("Saving the final dataset")
df.to_csv(output_path, sep="\t", index=False)
