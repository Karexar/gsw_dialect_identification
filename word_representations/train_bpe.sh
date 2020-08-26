#!/bin/bash

vocab_size=8000
model_name="bpe_8k"
model_path="../models/word_representations/bpe/${model_name}/"
input_file="../data/full_gsw_dataset/train.txt"
out_dir="../data/full_gsw_dataset/bpe"

echo "Creating directory path ${model_path}"
mkdir -p ${model_path}
mkdir -p ${out_dir}/${model_name}

# Put the root folder in the Python path in order to find modules
export PYTHONPATH=${PYTHONPATH}:"$(dirname "$(pwd)")"

python3.7 -u bpe.py \
      --model-path-bpe "../models/word_representations/bpe/bpe_8k/" \
      --model-name-bpe "bpe_8k" \
      --encode-file-bpe "../data/pmk/clean/train.txt" \
      --output-file-bpe "../data/pmk/bpe/bpe_8k/train.txt" \
      --encode-bpe 2>&1 | tee ${model_path}/train_${model_name}.log

#python3.7 -u bpe.py \
#      --input-file-bpe "../data/full_gsw_dataset/train.txt" \
#      --model-path-bpe "../models/word_representations/bpe/bpe_8k/" \
#      --model-name-bpe "bpe_8k" \
#      --vocab-size 8000 \
#      --encode-file-bpe "../data/full_gsw_dataset/train.txt" \
#      --output-file-bpe "../data/full_gsw_dataset/bpe/bpe_8k/train.txt" \
#      --train-bpe \
#      --encode-bpe 2>&1 | tee ${model_path}/train_${model_name}.log

#python3.7 -u bpe.py \
#      --input-file-bpe ${input_file} \
#      --model-path-bpe ${model_path} \
#      --model-name-bpe ${model_name} \
#      --vocab-size ${vocab_size} \
#      --encode-file-bpe ${input_file} \
#      --output-file-bpe ${out_dir}/${model_name}/train.txt \
#      --train-bpe \
#      --encode-bpe 2>&1 | tee ${model_path}/train_${model_name}.log
