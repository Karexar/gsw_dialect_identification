# This script merges the newly labelled dataset (i.e. unlabelled dataset we
# predicted) to the labelled dataset. The whole unlabelled dataset is merged to
# the labelled train set only. The test set will be the same as for the labelled
# dataset.

###  Settings  ################################################################
path_input_predicted = "data/twitter/unlabelled/labelled.csv"
dir_labelled = "data/twitter/labelled"
output_dir = "data/twitter/labelled_and_predicted/"
###############################################################################

import pandas as pd
import os
from pathlib import *
from shutil import copyfile

files = os.listdir(dir_labelled)
for name in ["train.csv", "test.csv"]:
    if name not in files:
        raise ValueError(f"{name} must be in 'dir_labelled' directory")

Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path_input_predicted, sep="\t", header=None)

train_path = os.path.join(dir_labelled, "train.csv")
new_train = pd.read_csv(train_path, sep="\t", header=None)
new_train = new_train.iloc[:, :2]

df = pd.concat([df, new_train]).sample(frac=1)
train_out_path = os.path.join(output_dir, "train.csv")
df.to_csv(train_out_path, sep="\t", header=None, index=False)

test_path = os.path.join(dir_labelled, "test.csv")
out_test_path = os.path.join(output_dir, "test.csv")
copyfile(test_path, out_test_path)
