import pandas as pd
import unicodedata
import json
import os
import re
from tqdm import tqdm
from utils.utils import *

from phrasal.norm_punc import *
from phrasal.mocy_splitter import MocySplitter
#from plato_ai_asr_preprocessor.preprocessor import Preprocessor
from preprocessing.cleaner import *

from enum import Enum
from sklearn.model_selection import train_test_split

from word_representations.bpe import BPE

##############################################################################
# Prepare the pmk dataset
##############################################################################

# ZH - Zürich: ZH, AG
# NW - North-West CH: BS, BL
# CE - Central Switzerland: OW, NW, LU, UR, SZ, ZG, GL
# GR - Grisons: GR
# BE - Bern: BE, SO
# VS - Valais: VS
# EA - Eastern Switzerland: AR, AI, SG, TG, SH
# RO - Fribourg: FR

UTTR_TRAIN = "data/pmk/gsw4.2.0_train/text.utterances"
LANG_TRAIN = "data/pmk/gsw4.2.0_train/utt2lang"
UTTR_DEV = "data/pmk/gsw4.2.0_dev/text.utterances"
LANG_DEV = "data/pmk/gsw4.2.0_dev/utt2lang"

# Minimum number of tokens in a sentence to keep it in the dataset
MIN_LEN_THRESH = 4
# Require a sentence to have at least this number of unique tokens
MIN_DIFF_TOKENS = 3

def clean_special_tokens(sentence):
    # remove foreign words and special labels from the given sentence
    special_chars = '[]-'
    clean_sentence = []
    for token in sentence.strip().split():
        if any(character in token for character in special_chars):
            continue
        clean_sentence.append(token)

    return ' '.join(clean_sentence)


def read_utterances(filename):
    # Read a pmk file, tab separated
    # column 0 = info and column 1 = sentence
    uttr_dict = {
        'uttr_id': [],
        'text': []
    }

    num_invalid_entries = 0
    num_short_entries = 0
    num_same_entries = 0
    # Two lines are wrong and contains the uuid as Text, we drop them
    drop_set = {"6798A89DC34993BC-m-60-ZH-yes-101_022",
                "75AEDEA671858A8E-m-60-CE-yes-103_003"}

    regexs = [
                (r'@\S*($|\s)', ' '), # mentions
                (r'#\S*($|\s)', ' '), # hashtags
                (r'https?[\w\.\:\/]*($|\s)', ' '), # links
             ]

    with open(filename, 'r', encoding='utf8') as file_in:
        for line in tqdm(file_in.readlines()):
            uttr_id, text = line.strip().split('\t')

            # drop wrong rows
            if text in drop_set:
                continue

            clean_text = Cleaner.preprocess(text, regexs)
            clean_text = normalize_text(clean_text)
            clean_text = clean_special_tokens(clean_text)
            clean_text = Cleaner.remove_special_chars(clean_text)
            clean_text = Cleaner.clean_spaces(clean_text)

            tokens = clean_text.split()
            if len(tokens) < MIN_LEN_THRESH:
                num_short_entries += 1
                # print(f'{line_processed[0]} has {len(tokens)} which is ' +
                #       f'less than the set min. threshold')
                continue

            if len(set(tokens)) < MIN_DIFF_TOKENS:
                num_same_entries += 1
                # print(f'{line_processed[0]} has {len(set(tokens))} which ' +
                #       f'is less than the required minimum of unique tokens')
                continue

            uttr_dict['uttr_id'].append(uttr_id)
            uttr_dict['text'].append(clean_text)

    print(f'Number of short sentences with less than {MIN_LEN_THRESH} ' +
          f'tokens: {num_short_entries}')
    print(f'Number of sentences with with less than {MIN_DIFF_TOKENS} unique ' +
          f'tokens: {num_same_entries}')
    valid_count = len(uttr_dict['uttr_id'])
    print(f'Number of valid sentences: {valid_count}')

    return uttr_dict


def read_dialect_info(filename):
    id_set = set()
    dialect_dict = {
        'uttr_id': [],
        'dialect': []
    }

    with open(filename, 'r', encoding='utf8') as file_in:
        for line in tqdm(file_in.readlines()):
            uttr_id_raw, dialect = line.strip().split('\t')
            uttr_id = '_'.join(uttr_id_raw.strip().split('_')[:2])
            if uttr_id in id_set:
                continue

            id_set.add(uttr_id)
            dialect_dict['uttr_id'].append(uttr_id)
            dialect_dict['dialect'].append(dialect)

    return dialect_dict


uttr_dict_train = read_utterances(UTTR_TRAIN)
dialect_dict_train = read_dialect_info(LANG_TRAIN)
print("train set : " + str(len(uttr_dict_train['uttr_id'])))
uttr_dict_dev = read_utterances(UTTR_DEV)
dialect_dict_dev = read_dialect_info(LANG_DEV)
print("dev set : " + str(len(uttr_dict_dev['uttr_id'])))

train_dev_df = pd.merge(pd.DataFrame.from_dict(uttr_dict_train),
                        pd.DataFrame.from_dict(dialect_dict_train),
                        how='inner',
                        on='uttr_id')
test_df = pd.merge(pd.DataFrame.from_dict(uttr_dict_dev),
                   pd.DataFrame.from_dict(dialect_dict_dev),
                   how='inner',
                   on='uttr_id')

print_set_sizes(train_dev_df, None, test_df)

# Creating a train, dev and test set

TEST_TRAIN_SPLIT = 0.1

class Dialect(str, Enum):
    ZH: str = 'ZH'  # Zürich
    NW: str = 'NW'  # North-West Switzerland
    CE: str = 'CE'  # Central Switzerland
    GR: str = 'GR'  # Graubünden / Grisons
    BE: str = 'BE'  # Bern
    VS: str = 'VS'  # Wallis / Valais
    EA: str = 'EA'  # Eastern Switzerland
    RO: str = 'RO'  # Fribourg

train_dev_df['label'] = 'unassigned'
print(train_dev_df.head())

# get the indices for the rows grouped by a source
# split the index lists into train, dev and test
# create a label for train, dev and test in the dataframe
# group by source and label
for dialect in Dialect:
    mask = train_dev_df['dialect'] == dialect.value
    group_indices = train_dev_df[mask].index.values.tolist()
    train, dev = train_test_split(group_indices,
                                  test_size = TEST_TRAIN_SPLIT,
                                  random_state=42)

    train_dev_df.at[train, 'label'] = 'train'
    train_dev_df.at[dev, 'label'] = 'dev'
    print(f'Split for {dialect}: {len(train)} train / {len(dev)} dev / ' +
          f'{len(group_indices)} total')

train_df = train_dev_df[train_dev_df['label'] == 'train'].drop(labels=['label'],
                                                               axis=1)
dev_df = train_dev_df[train_dev_df['label'] == 'dev'].drop(labels=['label'],
                                                           axis=1)

# Dropping Fribourg (not enough data in any dataset + big overlap with Bern)
train_df = train_df[train_df["dialect"]!="RO"]
dev_df = dev_df[dev_df["dialect"]!="RO"]
test_df = test_df[test_df["dialect"]!="RO"]

print_set_sizes(train_df, dev_df, test_df)

print(train_df.head())

# Save the dataset

DATA_BASE_FOLDER = 'data/pmk/clean'
TRAIN_DATA_CLASS = os.path.join(DATA_BASE_FOLDER, 'train.csv')
DEV_DATA_CLASS = os.path.join(DATA_BASE_FOLDER, 'dev.csv')
TEST_DATA_CLASS = os.path.join(DATA_BASE_FOLDER, 'test.csv')

def save_classification_data(dataset, filename):
    with open(filename, 'w', encoding='utf8') as file_out:
        for idx, row in tqdm(dataset.iterrows()):
            file_out.write(f"{row['text']}\t{row['dialect']}\n")

save_classification_data(train_df, TRAIN_DATA_CLASS)
save_classification_data(dev_df, DEV_DATA_CLASS)
save_classification_data(test_df, TEST_DATA_CLASS)

if False:
    bpe_model = BPE('not_used',
                    'models/word_representations/bpe/bpe_8k',
                    'bpe_8k',
                    training=False)

    def save_classification_data_bpe(dataset, filename, bpe_model):
        with open(filename, 'w', encoding='utf8') as file_out:
            for idx, row in tqdm(dataset.iterrows()):
                bpe_encoded = bpe_model.encode_sentence_2_pieces(row['text'])
                #file_out.write(f"{row['uttr_id']}\t{row['dialect']}\t" +
                #               f"{' '.join(bpe_encoded)}\n")
                file_out.write(f"{' '.join(bpe_encoded)}\t{row['dialect']}\n")

    DATA_BASE_FOLDER = 'data/pmk/bpe/bpe_8k'
    bpe_train_path = os.path.join(DATA_BASE_FOLDER, 'train.csv')
    save_classification_data_bpe(train_df, bpe_train_path, bpe_model)
    bpe_dev_path = os.path.join(DATA_BASE_FOLDER, 'dev.csv')
    save_classification_data_bpe(dev_df, bpe_dev_path, bpe_model)
    bpe_test_path = os.path.join(DATA_BASE_FOLDER, 'test.csv')
    save_classification_data_bpe(test_df, bpe_test_path, bpe_model)
