import pandas as pd
from naive_bayes.dialect_id import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
import os
import _thread
from utils.utils import *
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

###  Settings  #################################################################
DIR_PATH_DATASETS = ["data/archimob"]
DATASET_NAMES = ["archimob"]
WORD_NGRAMS = [1]
CHAR_NGRAMS = [1,2,3,4,5,6]
CLASSIFIER = MultinomialNB(alpha=0.01, fit_prior=False)
#CLASSIFIER = SVC(kernel="linear", probability=True)
train_vec = True
vec_dir = "models/naive_bayes/vectorizers"
train_clfs = True
clfs_dir = "models/naive_bayes/classifiers/archimob_ensemble_w1_c123456/"
################################################################################

_thread.start_new_thread( keep_alive, tuple() )

for i in range(len(DIR_PATH_DATASETS)):
    dir_path = DIR_PATH_DATASETS[i]
    print(f"Training for : {DATASET_NAMES[i]}")
    df_train = pd.read_csv(os.path.join(dir_path, "train.csv"),
                           sep="\t",
                           names=["Text", "Label"])
    print(f"Train set size : {df_train.shape[0]}")

    # Concat train and dev if available
    if "dev.csv" in os.listdir(dir_path):
        print("dev.csv file found, merging with train.csv...")
        df_dev = pd.read_csv(os.path.join(dir_path, "dev.csv"),
                             sep="\t",
                             names=["Text", "Label"])
        print(f"Dev set size : {df_dev.shape[0]}")
        df_train = pd.concat([df_train, df_dev], axis=0)
        print(f"Train+Dev set size : {df_train.shape[0]}")

    sentences = list(df_train["Text"])
    labels = list(df_train["Label"])


    # Compute the path of the vectorizer
    vec_name = f"{DATASET_NAMES[i]}_w"
    vec_name += ''.join([str(x) for x in WORD_NGRAMS]) + "_c"
    vec_name += ''.join([str(x) for x in CHAR_NGRAMS]) + ".joblib"
    vec_path = os.path.join(vec_dir, vec_name)
    Path(vec_dir).mkdir(parents=True, exist_ok=True)

    # Compute the path of the classifier
    clf_name = f"{DATASET_NAMES[i]}_w"
    clf_name += ''.join([str(x) for x in WORD_NGRAMS]) + "_c"
    clf_name += ''.join([str(x) for x in CHAR_NGRAMS]) + ".joblib"
    clf_path = os.path.join(clfs_dir, clf_name)
    Path(clfs_dir).mkdir(parents=True, exist_ok=True)

    # we give the paths if we want to load the models, otherwise the models
    # will be trained and save
    vec_path_init = None if train_vec else vec_path
    clfs_dir_init = None if train_clfs else clfs_dir
    dialect_id = Dialect_identification(vec_path=vec_path_init,
                                        clfs_dir=clfs_dir_init)

    # Train the vectorizer if needed
    if train_vec:
        print("Training the vectorizer")
        dialect_id.train_vectorizer(sentences,
                                    vec_path,
                                    word_ngrams=WORD_NGRAMS,
                                    char_ngrams=CHAR_NGRAMS)
    if train_clfs:
        print("Training the classifiers")
        for j in range(len(WORD_NGRAMS) + len(CHAR_NGRAMS)):
            print(j)
            dialect_id.train(sentences,
                             labels,
                             clf_path,
                             vect_idx=i,
                             clf=CLASSIFIER)

    # dialect_id.train(sentences,
    #                  labels,
    #                  clf_path,
    #                  clf=CLASSIFIER)


    # Make predictions on the test set
    df_test = pd.read_csv(os.path.join(dir_path, "test.csv"),
                          sep="\t",
                          names=["Text", "Label"])
    sentences = list(df_test["Text"])
    labels = list(df_test["Label"])
    preds = dialect_id.predict(sentences)

    print(f"Accuracy : {round(accuracy_score(labels, preds), 3)}")
    print(f"F1-micro : {round(f1_score(labels, preds, average='micro'), 3)}")
    print(f"F1-macro : {round(f1_score(labels, preds, average='macro'), 3)}")
