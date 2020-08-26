
python3.7 -u -m bert.bert_did \
      --pretrained_model_name "bert-base-german-cased" \
      --data_dir "data/twitter/labelled" \
      --model_dir "models/bert/twitter_german_cased_e3_g2_s146" \
      --cuda_device 0 \
      --sentence_length 146 \
      --fine_tune_model \
      --evaluate_model \
      --batch_size 32 \
      --epochs 3 \
      --learning_rate 0.00002 \
      --focal_loss_gamma 2 \


      # --data_dir "data/mix3" \
      #--label_names "BE,CE,EA,GR,NW,RO,VS,ZH"
      #--compute_max_length
      #--fine_tune_model \
      #--focal_loss_alpha 4.5 5.6 25.5 63.8 12.7 354.8 465.0 2.2
      # --focal_loss_alpha 0.9 1.0 1.0 1.0 1.0 1.0 0.6
      #--pretrained_model_name "bert-base-german-cased" \
      # --path_sentences "data/debug/test.csv" \

      #python -u -m bert.bert_did  --pretrained_model_name "bert-base-german-cased" --data_dir "data/twitter/labelled" --model_dir "models/bert/twitter_german_cased_e1_g2_s128_a2" --cuda_device 0 --sentence_length 128 --fine_tune_model  --evaluate_model --batch_size 32 --epochs 1 --learning_rate 0.00002 --focal_loss_gamma 2 --focal_loss_alpha 0.9 1.3 1.5 1.0 1.1 2.0 0.6

      # a1 : 0.9 1.3 1.5 1.0 1.1 2.0 0.6

#python -u -m bert.bert_did  --pretrained_model_name "bert-base-german-cased" --data_dir "data/twitter/labelled_and_predicted" --model_dir "models/bert/twitter_german_cased_mix_e3_g2_s128" --cuda_device 0 --sentence_length 128 --fine_tune_model --evaluate_model --batch_size 32 --epochs 3 --learning_rate 0.00002 --focal_loss_gamma 2

#--focal_loss_alpha 0.9 1.3 1.5 1.0 1.1 2.0 0.6
