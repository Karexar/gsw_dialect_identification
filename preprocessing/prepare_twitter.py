# This scripts load the twitter dataset and split it into a labelled and
# unlabelled set. It removes Fribourg because we have only very few data for
# this class.
# The labelled dataset will have 3 columns : sentence, dialect, user_id
# The unlabelled dataset will have 2 columns : sentence, user_id

###  Settings  #################################################################
dataset_path = "data/twitter/gsw_filtered.csv"
output_dir = "data/twitter"
################################################################################

import pandas as pd
from pathlib import Path
import os

Path(output_dir).mkdir(parents=True, exist_ok=True)
dir_labelled =  os.path.join(output_dir, "labelled")
dir_unlabelled = os.path.join(output_dir, "unlabelled")
Path(dir_labelled).mkdir(parents=True, exist_ok=True)
Path(dir_unlabelled).mkdir(parents=True, exist_ok=True)

print("Loading dataset")
df = pd.read_csv(dataset_path, sep="\t")
df = df[["sentence", "dialect", "user_id"]]

# Split the dataset into a labelled and unlabelled set
df_unlab = df[df["dialect"] == "Unknown"]
df_unlab = df_unlab.drop(columns=["dialect"])
df_lab = df[df["dialect"] != "Unknown"]
if df_unlab.shape[0] + df_lab.shape[0] != df.shape[0]:
    raise Exception(f"Wrong shapes : {df_unlab.shape[0]} + {df_lab.shape[0]} " +
                    f" != {df.shape[0]}")

# drop Fribourg
df_lab = df_lab[df_lab["dialect"]!="RO"]

print(f"Total length : {df.shape[0]}")
print(f"Length labelled : {df_lab.shape[0]}")
print(f"Length unlabelled : {df_unlab.shape[0]}")

# Save on disk
path = os.path.join(dir_labelled, "full.csv")
df_lab.to_csv(path, sep="\t", header=False, index=False)
path = os.path.join(dir_unlabelled, "full.csv")
df_unlab.to_csv(path, sep="\t", header=False, index=False)
