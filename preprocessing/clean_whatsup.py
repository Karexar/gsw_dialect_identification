import os
import re
import time
from utils.utils import *
from bert_lid import BertLid
import _thread
from twitter.tweet_filter import *
from twitter.geocoder import *
from phrasal.mocy_splitter import MocySplitter
from phrasal.pattern_sentence_filter import PatternSentenceFilter
from phrasal.norm_punc import normalize_text
import pandas as pd
import sys
from preprocessing.cleaner import *
import pathlib
from tqdm import tqdm



_thread.start_new_thread( keep_alive, tuple() )

##############################################################################
#  Settings
##############################################################################

input_dir = ["data/whatsup/raw",
             "data/whatsup/raw_demo",
             "data/whatsup/raw_dialog"]
output_path = "data/whatsup/gsw_sentences.csv"
lid_threshold = 0.95

##############################################################################
#  Load the files
##############################################################################
print("Loading files...")
lines = []
for dir_path in input_dir:
    for name in os.listdir(dir_path):
        with open(os.path.join(dir_path, name), "r") as f:
            raw_lines = f.readlines()
            res = [[name, line] for line in raw_lines]
            lines = lines + res

##############################################################################
# Parse the line to extract for each message the speaker id, chat id, message
# and message id
##############################################################################
# 0 : speaker line
# 1 : message lines
# 2 : message id line
print("Parsing messages...")
msgs = []
state = 0
i = 0
print()
while i < len(lines):
    print(str(i) + " / " + str(len(lines)), end="\r")
    # speaker line
    assert(lines[i][1].startswith("spk"))
    spk = lines[i][1]
    spk_id = re.search("spk(\d+).*", spk)[1]
    i+=1
    # message lines
    msg = ""
    while not lines[i][1].startswith("message ID: "):
        msg = msg + lines[i][1]
        i+=1
    # message id line
    assert(lines[i][1].startswith("message ID: "))
    msg_id = re.search("message ID: (\d+)", lines[i][1])[1]
    chat_id = lines[i][0]
    msgs.append([spk_id, chat_id, msg, msg_id])
    i+=1
print()
print(f"Messages count : {len(msgs)}")

##############################################################################
# Filter out encoded sentences (e.g. because of no permissions)
##############################################################################
texts_obj = [x for x in msgs if not x[2].startswith("redactedQ")]
texts_obj = [x for x in texts_obj if not x[2].startswith("systemQ")]
texts_obj = [x for x in texts_obj if not x[2].startswith("actionQ")]
texts_obj = [x for x in texts_obj if not x[2].startswith("mediaQ")]
texts_obj = [x for x in texts_obj if not x[2].startswith("emptyQ")]
texts_obj = [x for x in texts_obj if not x[2].startswith("encryptedQ")]
print(f"Text count : {len(texts_obj)}")
for x in texts_obj[:5]:
    print(x)
print("---")

##############################################################################
# Load the mapping between speaker and postcode
##############################################################################
df_spk = pd.read_csv("data/whatsup/speaker_to_postcode.csv",
                     sep="\t",
                     header=None)
df_spk.columns=["spk_id", "postcode"]
done = set(df_spk.spk_id.values)
done = set([str(x) for x in done])

final = []
for x in texts_obj:
    if not str(x[0]) in done:
        done.add(str(x[0]))
        final.append((int(x[0]), x[3]))

final = sorted(final, key=lambda x: x[0])
for x in final:
    print(x)

if len(final) > 0:
    raise Exception("Need more mapping")

##############################################################################
#  Clean the text by removing mentions, hashtags, and urls
#  Normalize the text
#  Split the text into sentences
#  Remove special characters
#  Clean spaces
##############################################################################

regexs = [("@\S*($|\s)", ""),
          ("#\S*($|\s)", ""),
          ("https?[\w\.\:\/]*($|\s)", "")]

print("Preprocessing the text...")
texts_obj = [[x[0], x[1], Cleaner.preprocess(x[2], regexs), x[3]]
             for x in tqdm(texts_obj)]
print("Normalizing the text...")
texts_obj = [[x[0], x[1], normalize_text(x[2], strip_emojis=True), x[3]]
             for x in tqdm(texts_obj)]
print("Splitting the text into sentences...")
splitter = MocySplitter()
texts_obj = [[x[0], x[1], splitter.split(x[2]), x[3]] for x in tqdm(texts_obj)]
texts_obj = [[x[0], x[1], y, x[3]] for x in tqdm(texts_obj) for y in x[2]]
print(f"  ==> {len(texts_obj)} sentences")
print("Removing special characters...")
texts_obj = [[x[0], x[1], Cleaner.remove_special_chars(x[2]), x[3]]
             for x in tqdm(texts_obj)]
print("Cleaning up spaces...")
texts_obj = [[x[0], x[1], Cleaner.clean_spaces(x[2]), x[3]]
             for x in tqdm(texts_obj)]

##############################################################################
#  Keep 'well-formed' sentences
##############################################################################

filterer = PatternSentenceFilter()
texts_obj = [x for x in tqdm(texts_obj) if filterer.is_valid(x[2])]

for x in texts_obj[:5]:
    print(x)
print("---")
print(f"Valid sentences : {len(texts_obj)}")

##############################################################################
#  Filter out non gsw sentences
##############################################################################

print("Filtering swiss-german sentences")
print("This may take a while...")
lid = BertLid()
preds = lid.predict_label([x[2] for x in texts_obj])
gsw_sentences = [texts_obj[i] for i in range(len(preds))
                 if preds[i] >= lid_threshold]

print(f"GSW sentences : {len(gsw_sentences)}")

##############################################################################
#  Map the speaker_id to the canton
##############################################################################

df = pd.DataFrame(gsw_sentences, columns=["speaker_id",
                                          "chat_id",
                                          "sentence",
                                          "message_id"])

df_spk = pd.read_csv("data/whatsup/speaker_to_postcode.csv",
                     sep="\t",
                     header=None)
df_spk.columns = ["spk_id", "postcode"]
df_spk = df_spk.set_index("spk_id")
spk_to_postcode = df_spk.to_dict()["postcode"]

df["postcode"] = df["speaker_id"].apply(lambda x: str(spk_to_postcode[int(x)]))
config = load_yaml("twitter/config.yaml")
geocoder = Geocoder(config)
postcodes = set(df["postcode"].values)
postcode_to_state = dict()

# todo : manual check
for postcode in postcodes:
    try:
        state = geocoder.postcode_to_state(postcode)
    except ApiException as e:
        if e.status == 404:
            print("Location not found, enter the corresponding state :")
            print("  0. None")
            for i in range(len(Geocoder.ch_states)):
                print(f"  {i+1}. {Geocoder.ch_states[i]}")
            loop_new_loc = True
            while loop_new_loc:
                state = input("Choose index of state: ")
                if (not state.isnumeric()
                    or int(state) < 0
                    or int(state) > 26):
                    print("Error : choose a number between 0 and 26")
                else:
                    loop_new_loc = False
                    keep_going = False
                    if state == 0:
                        state = None
                    else:
                        state = Geocoder.ch_states[int(state)-1]
        else:
            self.clean()
            raise(e)
    postcode_to_state[postcode] = state

df["state"] = df["postcode"].apply(lambda x: postcode_to_state[x])
df["dialect"] = df["state"].apply(Geocoder.state_to_dialect)
# select all lines where state is not None but dialect is None
# this is to check if the state is indeed non swiss-german
mask = (df["state"].isna()==False)&(df["dialect"].isna())
drop = set(df[mask]["state"].values)
print("Make sure none of the following are swiss-german state :")
print(drop)

##############################################################################
#  Split the dataset into labelled and unlabelled and save on disk
##############################################################################

# Create a directory for the labelled and unlabelled data
pathlib.Path('data/whatsup/labelled').mkdir(parents=True, exist_ok=True)
pathlib.Path('data/whatsup/unlabelled').mkdir(parents=True, exist_ok=True)

df = df[["sentence", "dialect", "speaker_id"]]
df_labelled = df[df["dialect"].isna()==False]
df_labelled = df_labelled.drop(columns=["speaker_id"])
df_unlabelled = df[df["dialect"].isna()]
df_unlabelled = df_unlabelled.drop(columns=["dialect"])

print(f"Sentence count : {df.shape[0]}")
print(df["dialect"].value_counts())

df_labelled.to_csv("data/whatsup/labelled/full.csv",
                   index=False,
                   header=None,
                   sep="\t")

df_unlabelled.to_csv("data/whatsup/unlabelled/full.csv",
                     index=False,
                     header=None,
                     sep="\t")
