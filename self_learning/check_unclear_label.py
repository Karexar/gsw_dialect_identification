# For each user, this script compute the average score for his dialect.
# For each dialect, the mapping between location to state is shown for the 10%
# users with this dialect that have the lowest average prediction score for
# this dialect. The idea is to check if the mapping was correct.

import pandas as pd
from utils.utils import *


print("Loading the datasets...")
df = pd.read_csv("data/twitter/labelled/full_predicted.csv", sep="\t")
df.user_id = df.user_id.astype(str)
tweets = load_obj("data/twitter/gsw_tweets.pkl")

print("Computing the unclear users")

dialects = sorted(list(set(df.label.values)))
unclear = []
for dialect in dialects:
    sub_df = df[df.label == dialect]
    scores = sub_df.groupby("user_id")[dialect].mean().sort_values()
    sep = len(scores) // 40
    unclear.extend(list(scores.index[:sep]))

user_to_location = dict()
for x in tweets:
    user_id = x[4]
    if user_id in unclear:
        user_to_location[user_id] = x[5]["user"]["location"]

user_to_dialect = dict()
for user_id in unclear:
    dialect = df[df.user_id==user_id]["label"].value_counts().index[0]
    user_to_dialect[user_id] = dialect

def ratio_twitter_loc(user):
    user_tweets = [x for x in tweets if x[4] == user]
    total = 0
    for x in user_tweets:
        if x[3] == "Twitter_place" or x[3] == "GPS":
           total += 1
    return total, len(user_tweets)

for user_id in unclear:
    twitter_loc, total_user_count = ratio_twitter_loc(user_id)
    print(f"{user_id} :  {user_to_location[user_id]}  =>  " +
          f"{user_to_dialect[user_id]}  " +
          f"({total_user_count} sentences)  " +
          f"({round(twitter_loc*100/total_user_count, 2)}% geo from twitter)")

print(f"Total unclear : {len(unclear)}")
