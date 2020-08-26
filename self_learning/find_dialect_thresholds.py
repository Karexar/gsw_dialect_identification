# This script takes the csv file created by predict_users.py, containing
# the sentences, labels, and user ids, and will find a threshold for each
# dialect, such that the dialect of a user can be infered.
# These threshold will be compared to the scores of a user for each dialect.
# The scores of a user correspond to the average of the sentence probabilities
# for each dialect, e.g.
# User 1
#              BE    ZH    SG
# sentence 1   0.1   0.2   0.7
# sentence 2   0.2   0.3   0.5
# score        0.15  0.25  0.6
#
# the scores for user 1 will be compared to a list of threshold corresponding to
# each dialect. If for a given dialect, the score of the user exceeds the
# threshold of the dialect, then all sentences of this users are labelled with
# this dialect.
# Example
#                    BE    ZH    SG
# User 1 score       0.15  0.25  0.6
# dialect threshold  0.2   0.4   0.5
#
# User 1 is predicted as having the dialect SG since 0.6 > 0.5
#
# The aim of this script is to find the dialect thresholds such that we can
# label users with a very high accuracy. The target accuracies are specific to
# each dialect because we may have a very low recall for a given class, maybe
# due to some wrong label.

# threshold found to keep 100% accuracy on training set and test set.
# note that we manually increased the VS threshold such that it gives 100%
# accuracy on the test set, while keeping 100% on the training set.
# thresholds = [0.6174, 0.4797, 0.2586, 0.4467, 0.433, 0.12667, 0.5745]


###############################################################################
###  Settings  ################################################################
###############################################################################
input_path = "data/twitter/labelled/full_predicted.csv"
# labels for accuracy are sorted : BE, CE, EA, GR, NW, VS, ZH
target_accuracies = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
min_sentence_per_user = 10
###############################################################################
###############################################################################
###############################################################################

import pandas as pd
import random

df = pd.read_csv(input_path, sep="\t")
label_names = list(df.columns[3:])

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

# create a train and test set by splitting users
users = list(set(df.user_id.values))
random.Random(42).shuffle(users)
sep = len(users)//10
user_test = users[:sep]
user_train = users[sep:]
mask = df.user_id.isin(user_train)
df_train = df[mask]
df_test = df[~mask]
print(f"train size : {df_train.shape[0]}")

# average the scores of each user for each dialect
df_user = df_train.groupby("user_id")[label_names].apply(lambda x: x.mean(axis=0))
df_true_label = df_train.groupby("user_id")["label"].apply(lambda x:
                                                    x.value_counts().index[0])
df_user = df_user.join(df_true_label, on="user_id")

thresholds = []
good_users = set()
for i, label in enumerate(label_names):
    print(label)
    df_user_dialect = df_user[df_user.label==label]
    min_threshold = df_user_dialect[label].min()
    #print(f"min threshold : {min_threshold}")
    dialect_users = df_user_dialect[df_user_dialect[label] >= min_threshold]
    total_users = df_user[df_user[label]>= min_threshold]
    accuracy = dialect_users.shape[0] / total_users.shape[0]
    #print(f"accuracy : {accuracy}")
    keep_going=True
    while accuracy < target_accuracies[i] and keep_going:
        min_threshold += 0.001
        dialect_users = df_user_dialect[df_user_dialect[label] >= min_threshold]
        total_users = df_user[df_user[label]>= min_threshold]
        if total_users.shape[0] == 0:
            keep_going = False
        else:
            accuracy = dialect_users.shape[0] / total_users.shape[0]

    if keep_going:
        thresholds.append(min_threshold)
        cur_count = dialect_users.shape[0]
        real_count = df_user_dialect.shape[0]
        ratio = round(cur_count*100 / real_count, 2)
        wrong_count = total_users.shape[0] - dialect_users.shape[0]
        wrong_users = set(total_users.index.values).difference(
                                            dialect_users.index.values)
        if len(wrong_users) > 0:
            print(total_users[total_users.index.isin(wrong_users)])
        print(f"Users labelled : {cur_count}/{real_count} ({ratio}%)" +
                f" (wrong: {wrong_count})")
        good_users = good_users.union(set(total_users.index.values))
        print(f"min threshold : {min_threshold}")
        print(f"accuracy : {accuracy}")
    else:
        threshold.append(None)

print([round(x, 4) for x in thresholds])
print(f"Users kept : {len(good_users)}")
print(f"Sentences kept : {df_train[df_train.user_id.isin(good_users)].shape[0]}")


# test set
# average the scores of each user for each dialect
df_user = df_test.groupby("user_id")[label_names].apply(lambda x: x.mean(axis=0))
df_true_label = df_test.groupby("user_id")["label"].apply(lambda x:
                                                    x.value_counts().index[0])
df_user = df_user.join(df_true_label, on="user_id")

def get_dominant_dialect(distrib):
    res = []
    for i in range(len(distrib)):
        if distrib.iloc[i] > thresholds[i]:
            res.append(label_names[i])
    if len(res) == 0 or len(res) > 1:
        return None
    return res[0]

df_user["dominant"] = df_user[label_names].apply(get_dominant_dialect, axis=1)
df_user_nonan = df_user.dropna(subset=["dominant"])
match = df_user_nonan["dominant"] == df_user_nonan["label"]
print(f"Accuracy on test set : {match.sum() / match.shape[0]}")
total_truely_label = 0
for label in label_names:
    print(label)
    # compute the accuracy : truely labelled over all labelled with the dialect
    df_user_dialect = df_user_nonan[df_user_nonan.label==label].copy()
    truely_labelled = (df_user_dialect.dominant == df_user_dialect.label).sum()
    total_truely_label += truely_labelled
    all_labelled = df_user_nonan[df_user_nonan.dominant==label].shape[0]
    print(f"Accuracy : {round(truely_labelled*100/all_labelled, 2)}%")
    # compute the coverage : truely labelled over all user with this dialect
    total_true = df_user[df_user.label==label].shape[0]
    ratio = round(truely_labelled*100/total_true, 2)
    print(f"Coverage : {truely_labelled}/{total_true} ({ratio}%)")
print(f"Total coverage : {total_truely_label/df_user.shape[0]}")




# df_user.name = "prediected_dialect"
# df_res = df.join(df_user, on="user_id")
# df_res = df_res.drop(columns=label_names+["user_id"])
#
# for dialect in label_names:
#     # get the lowest threshold
#     df_dialect = df_train[df_train.dialect==dialect][dialect]
#
#
# error = np.iinfo(np.int64).max
# cur_error = 0
# keep_going = True
# while keep_going:
#     print("*************************************************")
#     df_user = df.groupby("user_id")[label_names].apply(
#                     lambda x: dominant_dialect_function(x, ratio))
#     df_user.name = "prediected_dialect"
#     df_res = df.join(df_user, on="user_id")
#     df_res = df_res.drop(columns=label_names+["user_id"])
#
#     nan_count = df_res[df_res["dialect"].isna()].shape[0]
#     print(f"NaN percentage : {nan_count / df_res.shape[0]}")
#
#     df_nonan = df_res.dropna(subset=["dialect"])
#
#     # Compute the performance
#     true_labels = list(df_nonan["dialect"].values)
#     preds = list(df_nonan["dialect_predicted"].values)
#     score = f1_score(true_labels, preds, label_names, average="micro")
#     print(f"F1-micro : {score}")
#     score = f1_score(true_labels, preds, label_names, average="macro")
#     print(f"F1-macro : {score}")
#
#     # Compute the performance without ZH and BE, because we are interested in
#     # the less represented dialect.
#     print("Removing ZH and BE...")
#     mask = (df_nonan["dialect"] != "ZH") & (df_nonan["dialect"] != "BE")
#     df_nonan = df_nonan[mask]
#     print(df_nonan["dialect"].value_counts())
#     true_labels = list(df_nonan["dialect"].values)
#     preds = list(df_nonan["dialect_predicted"].values)
#
#     new_labels = label_names.copy()
#     new_labels.remove("ZH")
#     new_labels.remove("BE")
#     score = f1_score(true_labels, preds, new_labels, average="micro")
#     print(f"F1-micro : {score}")
#     score = f1_score(true_labels, preds, new_labels, average="macro")
#     print(f"F1-macro : {score}")
#
#     # we don't use value_counts because it would ignore label with 0 count
#     counts = df_res["dialect"].value_counts().sort_index()
#     if len(counts) != len(label_names):
#         counts_dict = dict()
#         for x in label_names:
#             if x in counts.index:
#                 counts_dict[x] = counts[x]
#             else:
#                 counts_dict[x] = 0
#         counts = pd.Series(counts_dict)
#
#     distrib = (counts / counts.sum()).round(4)
#     cur_error = 0
#     for i in range(len(distrib)):
#         print(f"{distrib.index[i]} : {distrib[i]}  ({true_distrib[i]})")
#         cur_error += (distrib[i] - true_distrib[i]) ** 2
#
#     print(f"Current error : {cur_error}")
#
#     if cur_error < error:
#         error = cur_error
#         # updating parameters
#         for i in range(len(ratio)):
#             diff = true_distrib[i] - distrib[i]
#             increment = diff*step
#             if distrib[i] / true_distrib[i] < 0.5:
#                 increment = np.sign(increment)*max_increment
#
#             ratio[i] += increment
#             ratio[i] = max(min(ratio[i], max_ratio), min_ratio)
#         print(f"New ratio : {[round(x, 3) for x in ratio]}")
#     else:
#         keep_going=False
