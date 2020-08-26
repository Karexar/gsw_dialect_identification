
python3.7 -u -m bert.bert_did \
      --pretrained_model_name "bert-base-german-cased" \
      --model_dir "models/bert/whatsup_german_cased_s146_g2" \
      --cuda_device 1 \
      --sentence_length 146 \
      --batch_size 32 \
      --path_sentences "data/debug/test.csv" \

      #--label_names "BE,CE,EA,GR,NW,RO,VS,ZH"
