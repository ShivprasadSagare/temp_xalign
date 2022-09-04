#!/usr/bin/env bash
export WANDB_API_KEY=bcb8767e750b1d80ba93361478ba51b615f2b8ce

for lang in en hi te bn pa or as gu mr kn ta ml;
do
    python3 train.py \
        --sanity_run yes \
        --train_path 'data/role_ordered_xalign_HRT_only/train.csv' \
        --val_path 'data/role_ordered_xalign_HRT_only/val.csv' \
        --test_path 'data/role_ordered_xalign_HRT_only/test.csv' \
        --tokenizer_name_or_path 'google/mt5-small' \
        --max_source_length 384 \
        --max_target_length 128 \
        --train_batch_size 20 \
        --val_batch_size 20 \
        --test_batch_size 20 \
        --model_name_or_path 'google/mt5-small' \
        --learning_rate 3e-5 \
        --eval_beams 4 \
        --tgt_max_seq_len 128 \
        --checkpoint_path None \
        --gpus 4 \
        --max_epochs 30 \
        --strategy 'ddp' \
        --log_dir 'experiments' \
        --project_name 'swft' \
        --run_name $lang+'_bilingual_finetuning_on-role_ordered_HRT_only_dataset' \
        --lang $lang
done