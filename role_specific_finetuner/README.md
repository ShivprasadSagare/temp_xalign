Steps for Manish Sir:

1. Set up the conda environment
    a. conda create --name role_spec
    b. install transformers through local cloned repo. cd into transformers and run pip install .
    c. install other dependencies pip install -r requirements.txt

2. Running the pretraining experiment
    a. pass appropriate train_file argument in run.sh file. Also, pass checkpoint_path=None.
    b. Run a sanity check using sanity_check=yes in run.sh.
    c. Run the run.sh file to start training.

3. Running the finetuning experiment
    a. Pass appropriate train_file arguments. Use the checkpoint created in above pretraining experiment, by setting chekpoint_path appropriately.
    b. Run the run.sh file to start training.