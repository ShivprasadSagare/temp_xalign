## temp_xalign
temporary repo for sharing code with Manish Sir.

> Tip: All the file paths below are w.r.t `role_specific_finetuner/`. Make sure you `cd` into it before running the scripts.

### Setting up environment and data
1. Setup the conda environment
```
conda create --name xalign_role python==3.7
conda activate xalign_role
pip install -r requirements.txt
cd transformers
pip install .
cd ..
```

2. Setup the data

2.1 Multilingual training experiment

wiki pretraining data: https://drive.google.com/file/d/1zvPuzPntsRtMisPfanAG_qH6oePKq_M0/view?usp=sharing <br>
xalign finetuning data: https://drive.google.com/file/d/1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki` and `./data/xalign_unified_script`.
```
mkdir data
cd data

gdown 1zvPuzPntsRtMisPfanAG_qH6oePKq_M0 
unzip wiki.zip

gdown 1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl
unzip xalign_unified_script.zip

cd ..
```

2.2 Only english training experiment

wiki pretraining data: https://drive.google.com/file/d/1JKPzlkfusE0rkpm3q0CBRc6cwSk2CzzD/view?usp=sharing <br>
xalign finetuning data: https://drive.google.com/file/d/1iEaYO0dE7owvX3m5suHEPZVctDpD5RQq/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_only_english` and `./data/xalign_only_english`.
```
mkdir data
cd data

gdown 1JKPzlkfusE0rkpm3q0CBRc6cwSk2CzzD
unzip wiki_only_english.zip

gdown 1iEaYO0dE7owvX3m5suHEPZVctDpD5RQq
unzip xalign_only_english.zip

cd ..
```

2.3 Multilingual pretraining experiment (increased data size)

wiki pretraining data: https://drive.google.com/file/d/1HTyF-IgeG1J-I-1Xn_gsRhKYEzDV31wy/view?usp=sharing <br>
xalign finetuning data: https://drive.google.com/file/d/1FR9Dpab5Z-hF0BM-IO1d-9MnElCNApaK/view?usp=sharing 

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_unified_script` and `./data/xalign_unified_script`.
```
mkdir data
cd data

gdown 1HTyF-IgeG1J-I-1Xn_gsRhKYEzDV31wy
unzip wiki_unified_script.zip

gdown 1FR9Dpab5Z-hF0BM-IO1d-9MnElCNApaK
unzip xalign_unified_script.zip

cd ..
```

2.4 Multilingual finetuning experiment (with ordered facts)

xalign finetuning data: [https://drive.google.com/file/d/1FR9Dpab5Z-hF0BM-IO1d-9MnElCNApaK/view?usp=sharing ](https://drive.google.com/file/d/1Yy4S0V1mRZ9nGHYhKevVDKLa4CcRlyuq/view?usp=sharing)


2.5 SWFT multilingual pretraining and then XAlign finetuning

<I>Before running this experiment, make sure you have standard recent installation of transformers, as opposed to custom edited version of transformers. For the same, uninstall existing transformers library, and install it again using `pip install transformers`. </I>


pretraining data: https://drive.google.com/file/d/18FdyPIR86wD8hvUYC-sU_BEO4WZeurXv/view?usp=sharing<br>
finetuning data: https://drive.google.com/file/d/1ancr8wa8gIWKrmOybi2UahAteaqFst_C/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_pretraining` and `./data/xalign_HRT_only`.
```
mkdir data
cd data

gdown 18FdyPIR86wD8hvUYC-sU_BEO4WZeurXv
unzip wiki_pretraining.zip

gdown 1ancr8wa8gIWKrmOybi2UahAteaqFst_C
unzip xalign_HRT_only.zip

cd ..
```
Make sure you are on the master branch.
Then run the file `run_both.sh` to run pretraining and finetuning sequentially automatically. Set `log_dir` and subsequent `checkpoint_path` correctly. Set other hyperparams as well before running.


2.6 KGPT multilingual pretraining and then XAlign finetuning

<I>Before running this experiment, make sure you have standard recent installation of transformers, as opposed to custom edited version of transformers. For the same, uninstall existing transformers library, and install it again using `pip install transformers`. </I>


pretraining data: <br>
finetuning data: 

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_pretraining` and `./data/xalign_HRT_only`.
```
mkdir data
cd data

gdown 
unzip wiki_pretraining.zip

gdown 
unzip xalign_finetuning.zip

cd ..
```
Then run the file `run_pretrain.sh` with proper hyperparams edited in it. Next, run `run_finetune.sh` with the checkpoint_path argument in it set to the checkpoint path created from previous pretraining. Set other hyperparams as well before running.

2.7 SWFT only_en pretraining and then XAlign finetuning

<I>Before running this experiment, make sure you have standard recent installation of transformers, as opposed to custom edited version of transformers. For the same, uninstall existing transformers library, and install it again using `pip install transformers`. </I>

pretraining data: https://drive.google.com/file/d/13Yb7eGuZ3-C3JdWQnFC0UepwQXbfGgJh/view?usp=sharing<br>
finetuning data: https://drive.google.com/file/d/1ancr8wa8gIWKrmOybi2UahAteaqFst_C/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_pretraining` and `./data/xalign_HRT_only`.
```
mkdir data
cd data

gdown 13Yb7eGuZ3-C3JdWQnFC0UepwQXbfGgJh
unzip wiki_pretraining.zip

gdown 1ancr8wa8gIWKrmOybi2UahAteaqFst_C
unzip xalign_HRT_only.zip

cd ..
```
Then run the file `run_pretrain.sh` with proper hyperparams edited in it. Next, run `run_finetune.sh` with the checkpoint_path argument in it set to the checkpoint path created from previous pretraining. Set other hyperparams as well before running.

2.8 XAlign finetuning with enhanced encoding strategy (role_specific encoding)
<I>Before running this experiment, make sure you have custom edited installation of transformers, as opposed to standard version of transformers. For the same, first switch the branch `role_ids`, then uninstall existing transformers library, and install it again using `cd transformers;pip install .`. </I>

finetuning data: https://drive.google.com/file/d/1JdtYxJlAYp7UGaD9mkY32ggehlPYcjJ4/view?usp=sharing

Then run the file `run_finetuning.sh` with proper hyperparams edited in it. Set the checkpoint path to None.


2.9 XAlign finetuning with copy_mechanism and role_specific encoding combined

Switch to the branch `role_ids_copy`.
<I>Before running this experiment, make sure you have custom edited installation of transformers, as opposed to standard version of transformers. For the same, first switch the branch `role_ids_copy`, then uninstall existing transformers library, and install it again using `cd transformers;pip install .`. </I>

finetuning data: https://drive.google.com/file/d/1R-PjPp2ylKKWkkoLxbJz3c_vxszBqsTv/view?usp=sharing

Then run the file `run_finetuning.sh` with proper hyperparams edited in it. Set the checkpoint path to None.


2.10 finetuning_HRTQRQT_multi_pretrained_ckpt_on_only_HRT_dataset

Git clone and stay on main branch. <I>Before running this experiment, make sure you have standard recent installation of transformers, as opposed to custom edited version of transformers. For the same, uninstall existing transformers library, and install it again using `pip install transformers`. </I> 

Download and unzip the data in data/ directory, using this link https://drive.google.com/file/d/1ancr8wa8gIWKrmOybi2UahAteaqFst_C/view?usp=sharing

Then , run `run_finetine.sh`, with appropriate parameters in it (epochs:30, checkpoint_path:'wandb'). Please keep checkpoint_path as 'wandb' only, exact location in the cloud is hardcoded in the script inside, we don't need to tweak it. 


2.11 finetuning_HRTQRQT_english_pretrained_ckpt_on_only_HRT_dataset

git clone and then switch to the branch `temp_en`. Rest all the instructions and data file is same as above experiment 2.10.

2.12 Combined instructions for running multi-stage and multi-task pretraining experiments.

Please switch to the `multi_stage_task_pretraining` branch for both the experiments.

Link to the data (same for both experiments): https://drive.google.com/file/d/1nokKQhGPe_3CvLGYH87pnyWvTD31ppyZ/view?usp=sharing

unzip the file data.zip and place `data` directory in `role_specific_finetuner/`. It has 4 subdirectories, `stage_1`, `stage_2`, `stage_1_and_2`, `stage_3`.

Please follow the files `run_multi_stage.sh` and `run_multi_task.sh` to start both experiments. Also, follow the data paths, --log_dir, and--checkpoint_path values in it carefully.

2.13 finetuning vanilla mT5 on xalign role-ordered dataset(HRT only) <br>
git clone the repo and stay on main branch.<br>

<I>Before running this experiment, make sure you have standard recent installation of transformers, as opposed to custom edited version of transformers. For the same, uninstall existing transformers library, and install it again using `pip install transformers`. </I> <br>

role_ordered_xalign_HRT_only dataset: https://drive.google.com/file/d/1_8B6qB_nvzCkRDW6hxfaSdKEJXE2Av5X/view?usp=sharing <br>
```
mkdir data
cd data

gdown 1_8B6qB_nvzCkRDW6hxfaSdKEJXE2Av5X
unzip role_ordered_xalign_HRT_only.zip
cd ..
```
run `run_finetune.sh` with proper arguments for batch size and GPUs.

2.14 finetuning role_specific mT5 on xalign role-ordered dataset(HRT only) <br>
git clone the repo and switch to `original_role_ids` branch <br>

<I>Before running this experiment, make sure you have custom edited installation of transformers, as opposed to standard version of transformers. For the same, first make sure you are on the branch `original_role_ids`, then uninstall existing transformers library, and install it again using `cd transformers;pip install .`. </I> <br>

setup the data similary as in above experiment 2.13. <br>
run `run_finetune.sh` with proper arguments for batch size and GPUs.

3. Some miscelleneous tips before running experiment
  * ~~Set up the wand api key using `export WANDB_API_KEY=bf6eddaca0cddb4d9e70aa37fb5ef56202d7ef74`~~
  * I have set up new wandb account for having more cloud storage. Please use this key starting August 9, 2022. Please type `export WANDB_API_KEY=bcb8767e750b1d80ba93361478ba51b615f2b8ce`
  * Make sure to sanity run the following scripts first, by passing `sanity_run=yes` in respective scripts. Later, pass `sanity_run=no` while running actual training.
  * Pass appropriate arguments for batch size, gpus, epochs, etc. in following scripts.
  * `conda activate xalign_role` from shell can throw error sometimes. Refer to initial few commented lines to reolve the issue instantly, if it occurs.
  
### Steps to run the pretraining experiment
* run `bash run_pretrain.sh`

### Steps to run the finetuning experiment
* set *checkpoint_path* variable in *run_finetune.sh*  to path of the checkpoint created from above pretraining experiment.
* run `bash run_finetune.sh`
