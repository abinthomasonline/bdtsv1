"""
Stage2: prior learning

run `python stage2.py`
"""
import os
import copy
from argparse import ArgumentParser
import argparse
import json

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Change relative imports to absolute imports
from .preprocessing.data_pipeline import build_custom_data_pipeline
from .preprocessing.preprocess import DatasetImporterCustom
from .experiments.exp_stage2 import ExpStage2
from .evaluation.evaluation import Evaluation
from .utils import get_root_dir, load_yaml_param_settings, str2bool

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

os.environ['WANDB_MODE'] = 'disabled'
os.environ["WANDB_DISABLE_PROGRESS_BAR"] = "True"


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--train_data_path', default='')
    parser.add_argument('--test_data_path', default='')
    parser.add_argument('--saved_models_dir', default='')
    # parser.add_argument('--static_cond_dim', default=1, type=int, help='Dimension of Static Conditions')
    # parser.add_argument('--seq_len', default=100, type=int, help='Length of sequence')
    # parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    # parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket')
    # parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage2(config: dict,
                 saved_models_dir: str,
                 dataset_name: str,
                 static_cond_dim: int,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 feature_extractor_type:str,
                 use_custom_dataset:bool,
                 ):
    project_name = 'TimeVQVAE-stage2'

    # fit
    # n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.TS.shape
    train_exp = ExpStage2(saved_models_dir, dataset_name, static_cond_dim, in_channels, input_length, config, feature_extractor_type, use_custom_dataset)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, 
                               config={**config, 'dataset_name': dataset_name, 'static_cond_dim': static_cond_dim, 'n_trainable_params': n_trainable_params, 'feature_extractor_type':feature_extractor_type})
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpu_device_ind

    if not os.path.isdir(saved_models_dir):
        os.mkdir(saved_models_dir)  

    # Define your early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val/loss',  # Metric to monitor; must match the name logged in validation_step
        min_delta=0.00,  # Minimum change in the monitored quantity to qualify as an improvement
        patience=10,  # Number of validation epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'  # Because we want to minimize the loss
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",  # Metric to monitor
        mode="min",  # Save the model with minimum 'val_loss'
        save_top_k=1, # Save the best model
        dirpath=saved_models_dir,  # Directory to save checkpoints
        filename=f'stage2',  # Custom filename format
        verbose=True
    )

    if config['early_stopping'] == True:
        callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback, early_stop_callback]
    else:
        callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback]
        
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=True,
                         callbacks=callbacks,
                         max_steps=config['trainer_params']['max_steps']['stage2'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
                         check_val_every_n_epoch=None,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    print('saving the model...')
    # if not os.path.isdir(saved_models_dir):
    #     os.mkdir(saved_models_dir)  
    # trainer.save_checkpoint(os.path.join(saved_models_dir, f'stage2-{dataset_name}.ckpt'))


    wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    else:
        config = load_yaml_param_settings(args.config)

    # config
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['dataset']['batch_sizes']['stage2']
    static_cond_dim = config['static_cond_dim']
    seq_len = config['seq_len']
    gpu_device_ind = config['gpu_device_id']
    saved_models_dir = args.saved_models_dir
    dataset_importer = DatasetImporterCustom(config=config, train_data_path=args.train_data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    if args.train_data_path and args.test_data_path:
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                           for kind in ['train', 'test']]
    if args.train_data_path and not args.test_data_path:
        train_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'train')
        test_data_loader = None

    if not args.train_data_path and args.test_data_path:
        train_data_loader = None
        test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')

    train_stage2(config, saved_models_dir, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
                 feature_extractor_type='rocket', use_custom_dataset=True)
