# This script split a dataset into a train and test set. It takes care keep the
# proportion for each class in the train and test set. The input dataset must
# have at least two columns, the first one being the sentence, and second one
# being the dialect.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

###  SETTINGS  ################################################################
input_dir = "data/twitter/labelled_and_predicted"
file_name = "full.csv"
delimiter = "\t"
header = None
output_dir = "data/twitter/labelled_and_predicted"
test_size = 0.1
###############################################################################

Path(output_dir).mkdir(parents=True, exist_ok=True)
# Load the source dataset
data_path = os.path.join(input_dir, file_name)
df = pd.read_csv(data_path, delimiter=delimiter, header=header)
df = df.iloc[:, 0:2]
df.columns = ["sentence", "label"]
print("Dataset shape : " + str(df.shape))
dialects = set(df["label"])

for dialect in dialects:
    print(dialect)
    group_indices = df[df["label"] == dialect].index.values.tolist()
    train, test = train_test_split(group_indices,
                                   test_size = test_size,
                                   random_state = 42)

    df.at[train, 'set_type'] = 'train'
    df.at[test, 'set_type'] = 'test'
    print(f'Split for {dialect}: {len(train)} train / {len(test)} test / ' +
          f'{len(group_indices)} total')

train_df = df[df['set_type'] == 'train'].drop(labels=['set_type'], axis=1)
test_df = df[df['set_type'] == 'test'].drop(labels=['set_type'], axis=1)
print("train_df shape : " + str(train_df.shape))
print("test_df shape : " + str(test_df.shape))

train_df.to_csv(os.path.join(output_dir, "train.csv"),
                sep="\t",
                index=False,
                header=None)
test_df.to_csv(os.path.join(output_dir, "test.csv"),
               sep="\t",
               index=False,
               header=None)
