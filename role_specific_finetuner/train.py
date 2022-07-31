from model.model import FineTuner 
from model.dataloader import DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import os
from datetime import datetime
from pathlib import Path
import wandb

def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = DataModule.add_datamodule_specific_args(parser)
    parser = FineTuner.add_model_specific_args(parser)
    # add miscellaneous arguments below
    parser.add_argument('--sanity_run', type=str, default='yes')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--log_dir', type=str, default='/scratch/shivprasad.sagare/experiments')
    parser.add_argument('--project_name', type=str, default='finetuner')
    parser.add_argument('--run_name', type=str, default='run_1')
    return parser

def main():
    parser = init_args()
    args = parser.parse_args()
    args = vars(args)

    dm = DataModule(
        train_path = args['train_path'],
        val_path = args['val_path'],
        test_path = args['test_path'],
        tokenizer_name_or_path = args['tokenizer_name_or_path'],
        max_source_length = args['max_source_length'],
        max_target_length = args['max_target_length'],
        train_batch_size = args['train_batch_size'],
        val_batch_size = args['val_batch_size'],
        test_batch_size = args['test_batch_size']
    )

    args.update({'tokenizer': dm.tokenizer})
    if args['checkpoint_path']=='None' or args['checkpoint_path']==None:
        model = FineTuner(**args)
    elif args['checkpoint_path']=='wandb':
        run = wandb.init()
        artifact = run.use_artifact('shivprasad/swft/model-28vrfbwv:v0', type='model')
        artifact_dir = artifact.download('/scratch/shivprasad.sagare/')
        # load checkpoint
        args['checkpoint_path'] = Path(artifact_dir) / "model.ckpt"
        model = FineTuner.load_from_checkpoint(**args)
    else:
        model = FineTuner.load_from_checkpoint(**args)

    os.makedirs(args['log_dir'], exist_ok=True)

    now = datetime.now()

    if args['sanity_run']=='yes':
        log_model = False
        limit_train_batches = 4
        limit_val_batches = 4
        limit_test_batches = 4
    else:
        log_model = True
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    logger = WandbLogger(
        name=args['run_name'] + '_' + now.strftime("%m/%d-%H%M"),
        project=args['project_name'],
        save_dir=args['log_dir'], 
        log_model=log_model
    )

    every_n_epochs = args['max_epochs'] // 5
    # checkpoint_callback_1 = ModelCheckpoint(every_n_epochs=every_n_epochs, save_top_k=-1)
    checkpoint_callback_2 = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=args['log_dir'], filename='model')
    trainer = pl.Trainer(
        gpus=args['gpus'], 
        max_epochs=args['max_epochs'], 
        strategy=args['strategy'], 
        logger=logger,
        callbacks=[checkpoint_callback_2],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches
    )

    # trainer.fit(model, dm)
    trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':
    main()

