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

wiki pretraining data: https://drive.google.com/file/d/16nmarTtpWquwVCOTp2uMDYz3NQ48Cb3W/view?usp=sharing <br>
xalign finetuning data: https://drive.google.com/file/d/1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it, and unzip the files into the newly created `./data` directory. Final directory structures should look like `./data/wiki` and `./data/xalign`.
```
mkdir data
cd data

gdown 16nmarTtpWquwVCOTp2uMDYz3NQ48Cb3W 
unzip wiki.zip

gdown 1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl
unzip xalign_unified_script.zip

cd ..
```

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
