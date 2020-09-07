# This scripts merges the hard known labels from the labelled dataset with the soft predicted labels from the unlabelled dataset.

###  Settings  #################################################################
#lab_path = "data/twitter/labelled_and_predicted/train.csv"
unlab_predicted_path = "data/twitter/unlabelled/full_predicted.csv"
#lab_test_path = "data/twitter/labelled/full_predicted.csv"
mix_path = "data/twitter/labelled_and_predicted/train.csv"
label_names = "BE,CE,EA,GR,NW,VS,ZH".split(',')
out_path = "data/twitter/soft_labels/hard_known_soft_preds/soft08/train.csv"
weight_soft = 0.8
################################################################################

import pandas as pd
import numpy as np

df_mix = pd.read_csv(mix_path, sep="\t", header=None)
#df_lab_pred = pd.read_csv(lab_path, sep="\t", header=None)
df_unlab_pred = pd.read_csv(unlab_predicted_path, sep="\t")

# remove all sentences from the unlabelled dataset that appear in the mix
# dataset
sentences = set(df_mix.iloc[:, 0].values)
df_unlab_pred = df_unlab_pred[~df_unlab_pred.sentence.isin(sentences)]

def map_hard_to_soft(hard_label):
    return ','.join(["1.0" if hard_label == label_names[i] else "0.0"
                    for i in range(len(label_names))])

# map the hard known labels to soft labels
df_mix.iloc[:, 1] = df_mix.iloc[:, 1].map(map_hard_to_soft)
df_mix.columns = ["sentence", "label"]

def interpolate_soft_hard(logits, weight_soft):
    hard = np.array([1 if i == np.argmax(logits) else 0
                     for i in range(len(logits))])
    return list(logits*weight_soft + hard*(1.0-weight_soft))

# map the predicted soft labels to str
print(df_unlab_pred.iloc[0])
df_unlab_pred["label"] = df_unlab_pred.iloc[:, 2:].apply(lambda x:
    ','.join([str(y) for y in interpolate_soft_hard(x, weight_soft)]), axis=1)
df_unlab_pred = df_unlab_pred[["sentence", "label"]]
print(df_unlab_pred.iloc[0])

res = pd.concat([df_mix, df_unlab_pred])
res = res.sample(frac=1.0) # shuffle

res.to_csv(out_path, sep="\t", header=None, index=False)
