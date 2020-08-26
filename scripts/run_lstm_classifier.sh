epochs=20
algo="random"
embed_size=650
hid_layers=800
dataset_name="pmk"

model_path_lstm="./models/lstm/${algo}_embed_${embed_size}_nhid_${hid_layers}_${dataset_name}"

mkdir -p ${model_path_lstm}

python3.7 -u -m lstm.main_dialect_classifier \
      --cuda \
      --cuda-device "cuda:1" \
      --num_hidden ${hid_layers} \
      --epochs ${epochs} \
      --data "data/${dataset_name}/bpe/bpe_8k" \
      --use-pretrained-embed \
      --model-path-lstm "${model_path_lstm}/model.pt" \
      --eval-lstm \
      --log_interval 100 \
      --batch_size 10 \
      --bptt 80 \
      --dropout 0.25 \
      --eval_batch_size 5 \
      --learning_rate 1 \
      --clip 0.4 \
      --num_layers 2 \
      --embed-size ${embed_size} ${algo} 2>&1 | tee ${model_path_lstm}/train.log
