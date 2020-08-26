import pandas as pd
from tqdm import tqdm
from word_representations.bpe import BPE
from sklearn.model_selection import train_test_split
import os
from utils.utils import *
from pathlib import Path

###  Settings  ################################################################
dir_path = "data/mix4"
dev_proportion = 0.1
model_name = "bpe_8k"
###############################################################################

# Load the train and valid set
df_train_dev = pd.read_csv(os.path.join(dir_path, "train.csv"),
                           names=["Text", "Label"],
                           delimiter="\t")
df_valid = None
if "dev.csv" in os.listdir(dir_path):
    df_valid = pd.read_csv(os.path.join(dir_path, "dev.csv"),
                                        names=["Text", "Label"],
                                        delimiter="\t")
elif "valid.csv" in os.listdir(dir_path):
    df_valid = pd.read_csv(os.path.join(dir_path, "valid.csv"),
                                        names=["Text", "Label"],
                                        delimiter="\t")
if df_valid is not None:
    df_train_dev = pd.concat([df_train_dev, df_valid])

# Load the test set
df_test = pd.read_csv(os.path.join(dir_path, "test.csv"),
                      names=["Text", "Label"],
                      delimiter="\t")

# Creating a train, dev and test set
dialects = set(df_train_dev["Label"])

df_train_dev['set_type'] = 'unassigned'
for dialect in dialects:
    mask = df_train_dev['Label'] == dialect
    group_indices = df_train_dev[mask].index.values.tolist()
    train, dev = train_test_split(group_indices,
                                  test_size = dev_proportion,
                                  random_state=42)

    df_train_dev.at[train, 'set_type'] = 'train'
    df_train_dev.at[dev, 'set_type'] = 'dev'
    print(f'Split for {dialect}: {len(train)} train / {len(dev)} dev / ' +
          f'{len(group_indices)} total')

mask = df_train_dev['set_type'] == 'train'
df_train = df_train_dev[mask].drop(labels=['set_type'], axis=1)
mask = df_train_dev['set_type'] == 'dev'
df_dev = df_train_dev[mask].drop(labels=['set_type'], axis=1)

print_set_sizes(df_train, df_dev, df_test)

bpe_model = BPE("not_used",
                os.path.join("models/word_representations/bpe", model_name),
                model_name,
                training=False)

def save_classification_data_bpe(df, filename, bpe_model):
    with open(filename, 'w', encoding='utf8') as file_out:
        for idx, row in tqdm(df.iterrows()):
            bpe_encoded = bpe_model.encode_sentence_2_pieces(row['Text'])
            file_out.write(f"{' '.join(bpe_encoded)}\t{row['Label']}\n")

out_dir_path = os.path.join(os.path.join(dir_path, "bpe"), model_name)
Path(out_dir_path).mkdir(parents=True, exist_ok=True)
for e in [[df_train, "train.csv"], [df_dev, "dev.csv"], [df_test, "test.csv"]]:
    path = os.path.join(out_dir_path, e[1])
    save_classification_data_bpe(e[0], path, bpe_model)
