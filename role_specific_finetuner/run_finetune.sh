#Activate conda environment xalign_role.
#Following line is added to resolve the error 'Your shell has not been properly configured to use 'conda activate'. 
#Uncomment and Pass the appropriate anaconda/miniconda path in the below line, if you face the error.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate xalign_role

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
checkpoint_path=None

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--sanity_run yes \
--train_path 'data/xalign_unified_script/train.csv' \
--val_path 'data/xalign_unified_script/val.csv' \
--test_path 'data/xalign_unified_script/test.csv' \
--tokenizer_name_or_path 'google/mt5-base' \
--max_source_length 384 \
--max_target_length 128 \
--train_batch_size 20 \
--val_batch_size 20 \
--test_batch_size 20 \
--model_name_or_path 'google/mt5-base' \
--learning_rate 3e-5 \
--eval_beams 4 \
--tgt_max_seq_len 128 \
--checkpoint_path $checkpoint_path \
--gpus 4 \
--max_epochs 30 \
--strategy 'ddp' \
--log_dir 'experiments' \
--project_name 'swft' \
--run_name 'best_model_large_variant'

