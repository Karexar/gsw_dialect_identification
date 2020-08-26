# This script label all users from an unlabelled dataset using the thresholds
# created by the find_dialect_thresholds script


###  Settings  ################################################################
input_path = "data/twitter/unlabelled/full_predicted.csv"
output_labelled = "data/twitter/unlabelled/labelled.csv"
output_unlabelled = "data/twitter/unlabelled/unlabelled.csv"
# threshold index : BE, CE, EA, GR, NW, VS, ZH
thresholds = [0.6174, 0.4797, 0.2586, 0.4467, 0.433, 0.12667, 0.5745]
min_sentence_per_user = 10
###############################################################################

import pandas as pd

df = pd.read_csv(input_path, sep="\t")
label_names = list(df.columns[2:])

# drop users with too few sentences
print(f"Dataset size : {df.shape[0]}")
tmp = len(set(df["user_id"].values))
print(f"Users : {tmp}")
counts = df["user_id"].value_counts()
good_users = set(counts[counts >= min_sentence_per_user].index)
df = df[df.user_id.isin(good_users)]
print(f"Dataset filtered : {df.shape[0]}")
tmp = len(set(df["user_id"].values))
print(f"Users : {tmp}")

# average the scores of each user for each dialect
df_user = df.groupby("user_id")[label_names].apply(lambda x: x.mean(axis=0))

def get_dominant_dialect(distrib):
    res = []
    for i in range(len(distrib)):
        if distrib.iloc[i] > thresholds[i]:
            res.append(label_names[i])
    if len(res) == 0 or len(res) > 1:
        return "Unknown"
    return res[0]

df_user["label"] = df_user[label_names].apply(get_dominant_dialect, axis=1)
df_user = df_user.drop(columns=label_names)
df = df.join(df_user, on="user_id").drop(columns=label_names+["user_id"])

unlab = df[df.label=="Unknown"]
lab = df[df.label!="Unknown"]

print(f"Labelled : {lab.shape[0]}")
print(f"Unlabelled : {unlab.shape[0]}")

print(lab.label.value_counts())

unlab.to_csv(output_unlabelled, sep="\t", header=None, index=False)
lab.to_csv(output_labelled, sep="\t", header=None, index=False)
