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
from preprocessing.data_pipeline import build_custom_data_pipeline
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from preprocessing.preprocess import DatasetImporterCustom

from experiments.exp_stage2 import ExpStage2
from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, str2bool


os.environ['WANDB_MODE'] = 'disabled'

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_names', nargs='+', help="e.g., Adiac Wafer Crop`.", default='')
    parser.add_argument('--train_data_path', default='')
    parser.add_argument('--test_data_path', default='')
    parser.add_argument('--static_cond_dim', default=1, type=int, help='Dimension of Static Conditions')
    parser.add_argument('--seq_len', default=100, type=int, help='Length of sequence')
    parser.add_argument('--gpu_device_ind', nargs='+', default=[0], type=int, help='Indices of GPU devices to use.')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage2(config: dict,
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
    train_exp = ExpStage2(dataset_name, static_cond_dim, in_channels, input_length, config, feature_extractor_type, use_custom_dataset)
    
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

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
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
    if not os.path.isdir(get_root_dir().joinpath('saved_models')):
        os.mkdir(get_root_dir().joinpath('saved_models'))
    trainer.save_checkpoint(os.path.join(f'saved_models', f'stage2-{dataset_name}.ckpt'))


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
    dataset_importer = DatasetImporterCustom(train_data_path=args.train_data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                           for
                                           kind in ['train', 'test']]
    train_stage2(config, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
                 feature_extractor_type='rocket', use_custom_dataset=True)
