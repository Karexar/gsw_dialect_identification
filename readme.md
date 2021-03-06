# Dialect identification for Swiss-German

This repository contains a BERT model to identify Swiss-German dialects. We use the 'bert-base-german-cased' pretrained model and fine-tune it on labelled Swiss-German sentences. Labelled GSW sentences can be obtained for example using the Twitter streamer (https://github.com/Karexar/twitter_streamer). The dialect is inferred from the Swiss state ("canton") the Twitter user lives in, when this information is available.

## Installation

### typechecker

To check argument types at runtime, a typechecker is used. This needs to be installed manually from

https://github.com/Karexar/typechecker

### other modules

```zsh
pip install -r requirements.txt
```

If you encounter the following error :
```zsh
Could not find a version that satisfies the requirement torch>=0.4.1
```
Try to install torch by specifying the link :
```zsh
python -m pip install torch==1.6 -f https://download.pytorch.org/whl/torch_stable.html
```
Note that any torch version up to 1.6 should work.

## Setup

First prepare a labelled dataset with dialects. If you use the twitter dataset, you can run the following scripts. Do not forget to update the settings in each script.

```zsh
python -m preprocessing.prepare_twitter
python -m preprocessing.split_train_test
```

## Fine-tuning

Then after updating the parameter you can run :
```zsh
./scripts/run_bert.sh
```

## Self-learning

If we know the users for each sentence, we can perform semi-supervised learning using the unlabelled sentences of the dataset. The first step is to predict the dialect for all users of the labelled dataset (i.e. where we already know the true dialect for each user). This is to find the thresholds such that we can predict dialect with a high confidence. Check the parameters to make sure you predict on the labelled dataset.

```zsh
python -m self_learning.predict_users
python -m self_learning.find_dialect_thresholds
```

Then you can make predictions on the unlabelled dataset, then use the thresholds to label the users, and finally merge the new dataset with the training set of the labelled dataset. Check the parameters for each file.

```zsh
python -m self_learning.predict_users
python -m self_learning.label_users
python -m self_learning.merge_predicted
```

Now you can fine-tune the model with the new data.

## Soft-labels

Swiss-German dialects are hard to differentiate. A classification based on the Swiss state the user lives in is not very accurate. We may have people living and working at different places, or that moved on to somewhere else. In either case, there is not only one dialect for this user. Soft-label are used in the following way. First we predict the dialect for all sentences in the dataset and we keep the logits.

```zsh
python -m soft_labels.predict_sentences
```

Then we fine-tune the model on the new dataset using the options --soft_label and --weight_soft <weight>. the first is to indicate we want to fine-tune using soft-labels. The second is to indicate the weight to give to soft label, with respect to hard label. A value of 1 indicates we take only the soft labels. 0 will produce the same result as with hard labels. A value in-between will compute an interpolation between the soft labels and hard label.

The classifier can be slightly improved by fine-tuning on the combination of known labels as hard labels (e.g. [0, 1, 0, 0, 0, 0, 0]) and the predicted labels as soft labels (e.g. [0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0]) with a weight of 0.8 for soft labels. 


## Notes

You may need to install additional package if you want to try the AWD-LSTM. It was not included in the requirements.txt file because BERT is the best model for this task.

When installing glove-python with python3.7, you may encounter this error :
```
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```
In this case refer to https://github.com/maciejkula/glove-python/issues/96. Basically you will need to download the repository and recompiling some files

```
cython glove_cython.pyx
cythonize glove_cython.pyx

cython metrics/accuracy_cython.pyx
cythonize metrics/accuracy_cython.pyx

cython --cplus corpus_cython.pyx
cd ..
python setup.py cythonize
make
```

If there is an error with a missing Python.h, you may need to run :

```
sudo apt-get install python3.7-dev
```
