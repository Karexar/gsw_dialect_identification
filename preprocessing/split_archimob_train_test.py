import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the source dataset
dir_path = "data/archimob"
df = pd.read_csv(os.path.join(dir_path, "archimob.csv"))
print("dataset shape : " + str(df.shape))
dialects = set(df["Label"])

for dialect in dialects:
    group_indices = df[df['Label'] == dialect].index.values.tolist()
    train, test = train_test_split(group_indices,
                                   test_size = 0.1,
                                   random_state = 42)

    df.at[train, 'set_type'] = 'train'
    df.at[test, 'set_type'] = 'test'
    print(f'Split for {dialect}: {len(train)} train / {len(test)} test / ' +
          f'{len(group_indices)} total')

train_df = df[df['set_type'] == 'train'].drop(labels=['set_type'], axis=1)
test_df = df[df['set_type'] == 'test'].drop(labels=['set_type'], axis=1)
print("train_df shape : " + str(train_df.shape))
print("test_df shape : " + str(test_df.shape))

train_df.to_csv(os.path.join(dir_path, "train.csv"),
                sep="\t",
                index=False,
                header=None)
test_df.to_csv(os.path.join(dir_path, "test.csv"),
               sep="\t",
               index=False,
               header=None)
