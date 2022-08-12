#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path=None

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run yes \
--train_path 'data/stage_1_and_2/train.csv' \
--val_path 'data/stage_1_and_2/val.csv' \
--test_path 'data/stage_1_and_2/test.csv' \
--tokenizer_name_or_path 'google/mt5-small' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 1 \
--val_batch_size 1 \
--test_batch_size 1 \
--model_name_or_path 'google/mt5-small' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 2 \
--max_epochs 3 \
--strategy 'ddp' \
--log_dir 'experiments_stage_1_and_2' \
--project_name 'swft' \
--run_name 'stage_1_and_2_pretraining'

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path='experiments_stage_1_and_2/model.ckpt'

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run yes \
--train_path 'data/stage_3/train.csv' \
--val_path 'data/stage_3/val.csv' \
--test_path 'data/stage_3/test.csv' \
--tokenizer_name_or_path 'google/mt5-small' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 1 \
--val_batch_size 1 \
--test_batch_size 1 \
--model_name_or_path 'google/mt5-small' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 2 \
--max_epochs 3 \
--strategy 'ddp' \
--log_dir 'experiments_stage_3_multitask' \
--project_name 'swft' \
--run_name 'stage_3_finetuning_multitask'
