"""
Stage 1: VQ training

run `python stage1.py`
"""
import os
from argparse import ArgumentParser
import copy
import json

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .preprocessing.preprocess import DatasetImporterCustom
from .experiments.exp_stage1 import ExpStage1
from .preprocessing.data_pipeline import build_custom_data_pipeline
from .utils import get_root_dir, load_yaml_param_settings, str2bool

os.environ['WANDB_MODE'] = 'disabled'
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
    # parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()


def train_stage1(config: dict,
                 saved_models_dir: str,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_device_ind,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'TimeVQVAE-stage1'

    # fit
    # _, in_channels, input_length = train_data_loader.dataset.TS.shape
    in_channels = len(train_data_loader.dataset.TS.columns)
    size = train_data_loader.dataset.TS.size()
    input_length = size[size.index[0]]
    train_exp = ExpStage1(in_channels, input_length, config)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_name': dataset_name, 'n_trainable_params:': n_trainable_params})

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
    # Initialize ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",  # Metric to monitor
        mode="min",  # Save the model with minimum 'val_loss'
        save_top_k=1, # Save the best model
        dirpath=saved_models_dir,  # Directory to save checkpoints
        filename=f'stage1-{dataset_name}',  # Custom filename format
        verbose=True
    )

    if config['early_stopping'] == True:
        callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback, early_stop_callback]
    else:
        callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback]
        
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=callbacks,
                         max_steps=config['trainer_params']['max_steps']['stage1'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
                         check_val_every_n_epoch=None,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    # if not os.path.isdir(saved_models_dir):
    #     os.mkdir(saved_models_dir)
    # trainer.save_checkpoint(os.path.join(saved_models_dir, f'stage1-{dataset_name}.ckpt'))


if __name__ == '__main__':
    # load config
    args = load_args()
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    else:
        config = load_yaml_param_settings(args.config)
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['dataset']['batch_sizes']['stage1']
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
        
    train_stage1(config, saved_models_dir, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)


