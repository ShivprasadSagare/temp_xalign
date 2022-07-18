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
finetuning data: https://drive.google.com/file/d/1JdtYxJlAYp7UGaD9mkY32ggehlPYcjJ4/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki_pretraining` and `./data/xalign_finetuning`.
```
mkdir data
cd data

gdown 18FdyPIR86wD8hvUYC-sU_BEO4WZeurXv
unzip wiki_pretraining.zip

gdown 1JdtYxJlAYp7UGaD9mkY32ggehlPYcjJ4
unzip xalign_finetuning.zip

cd ..
```
Then run the file `run_pretrain.sh` with proper hyperparams edited in it. Next, run `run_finetune.sh` with the checkpoint_path argument in it set to the checkpoint path created from previous pretraining. Set other hyperparams as well before running.

3. Some miscelleneous tips before running experiment
  * Set up the wand api key using `export WANDB_API_KEY=bf6eddaca0cddb4d9e70aa37fb5ef56202d7ef74`
  * Make sure to sanity run the following scripts first, by passing `sanity_run=yes` in respective scripts. Later, pass `sanity_run=no` while running actual training.
  * Pass appropriate arguments for batch size, gpus, epochs, etc. in following scripts.
  * `conda activate xalign_role` from shell can throw error sometimes. Refer to initial few commented lines to reolve the issue instantly, if it occurs.
  
### Steps to run the pretraining experiment
* run `bash run_pretrain.sh`

### Steps to run the finetuning experiment
* set *checkpoint_path* variable in *run_finetune.sh*  to path of the checkpoint created from above pretraining experiment.
* run `bash run_finetune.sh`
