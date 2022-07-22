#!/bin/bash
#Activate conda environment xalign_role.
#Following line is added to resolve the error 'Your shell has not been properly configured to use 'conda activate'. 
#Uncomment and Pass the appropriate anaconda/miniconda path in the below line, if you face the error.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path=None

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run yes \
--train_path 'data/wiki_pretraining/train.csv' \
--val_path 'data/wiki_pretraining/val.csv' \
--test_path 'data/wiki_pretraining/test.csv' \
--tokenizer_name_or_path 'google/mt5-small' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 2 \
--val_batch_size 2 \
--test_batch_size 3 \
--model_name_or_path 'google/mt5-small' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 2 \
--max_epochs 5 \
--strategy 'ddp' \
--log_dir 'experiments' \
--project_name 'swft' \
--run_name 'multilingual_pretraining'

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path='experiments/model.ckpt'

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run yes \
--train_path 'data/xalign_finetuning/train.csv' \
--val_path 'data/xalign_finetuning/val.csv' \
--test_path 'data/xalign_finetuning/test.csv' \
--tokenizer_name_or_path 'google/mt5-small' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 4 \
--val_batch_size 4 \
--test_batch_size 4 \
--model_name_or_path 'google/mt5-small' \
--learning_rate 1e-3 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 2 \
--max_epochs 5 \
--strategy 'ddp' \
--log_dir '/scratch/experiments' \
--project_name 'xalign' \
--run_name 'finetuning'

