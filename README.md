## temp_xalign
temporary repo for sharing code with Manish Sir.

### Setting up environment and data
1. Setup the conda environment
```
conda create --name xalign_role python==3.7
pip install -r requirements.txt
cd transformers
pip install .
```

2. Setup the data

wiki pretraining data: https://drive.google.com/file/d/16nmarTtpWquwVCOTp2uMDYz3NQ48Cb3W/view?usp=sharing <br>
xalign finetuning data: https://drive.google.com/file/d/1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl/view?usp=sharing

Alternatively, you can use following gdown commands to directly download it.
```
cd data
gdown 16nmarTtpWquwVCOTp2uMDYz3NQ48Cb3W 
gdown 1xuS8zvq4k2F6Uxfx6XF6R3ueKNNFodjl
```
unzip above files into the `./data` directory.. Final directory structures should look like `./data/wiki` and `./data/xalign`.

3. Some miscelleneous tips before running experiment
  * Set up the wand api key using `export WANDB_API_KEY=bf6eddaca0cddb4d9e70aa37fb5ef56202d7ef74`
  * Make sure to sanity run the following scripts first, by passing `sanity_run=yes` in respective scripts. Later, pass `sanity_run=no` while running actual training.
  
### Steps to run the pretraining experiment
* run `bash run_pretrain.sh`

### Steps to run the finetuning experiment
* set *checkpoint_path* variable in *run_finetune.sh*  to path of the checkpoint created from above pretraining experiment.
* run `bash run_finetune.sh`