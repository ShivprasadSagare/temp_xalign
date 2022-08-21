#!/bin/bash
#SBATCH -A research
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=3G

#Activate conda environment role_spec
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export WANDB_API_KEY=bcb8767e750b1d80ba93361478ba51b615f2b8ce

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path=wandb

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run no \
--train_path 'data/stage_3/train.csv' \
--val_path 'data/stage_3/val.csv' \
--test_path 'data/stage_3/test.csv' \
--tokenizer_name_or_path 'google/mt5-small' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 4 \
--val_batch_size 4 \
--test_batch_size 4 \
--model_name_or_path 'google/mt5-small' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 1 \
--max_epochs 2 \
--strategy 'ddp' \
--log_dir 'experiments' \
--project_name 'swft' \
--run_name 'predicting_multi_task__pretraining_corrected_expt'

