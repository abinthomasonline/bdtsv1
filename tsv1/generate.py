"""
Stage2: prior learning

run `python stage2.py`
"""
import os
import argparse
from argparse import ArgumentParser
from typing import Union
import random
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .preprocessing.data_pipeline import build_custom_data_pipeline
from .preprocessing.preprocess import DatasetImporterCustom
import pandas as pd
import json

from .evaluation.evaluation import Evaluation
from .utils import get_root_dir, load_yaml_param_settings, str2bool

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
    parser.add_argument('--gpu_device_idx', default=0, type=int)
    parser.add_argument('--use_fidelity_enhancer', type=str2bool, default=False, help='Use the fidelity enhancer')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=False, help='Using a custom dataset, then set it to True.')
    parser.add_argument('--saved_models_dir', default='')
    return parser.parse_args()


def generate_data(config: dict,
             saved_models_dir: str,
             dataset_name: str,
             static_cond_dim: int,
             static_conditions,
             gpu_device_ind,
             use_fidelity_enhancer:bool,
             feature_extractor_type:str,
             use_custom_dataset:bool=False,
             rand_seed:Union[int,None]=None,
             ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    if not isinstance(rand_seed, type(None)):
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)

    in_channels, input_length = config['dataset']['num_features'], config['seq_len']
    
    # wandb init
    wandb.init(project='TimeVQVAE-evaluation', 
               config={**config, 'dataset_name': dataset_name, 'static_cond_dim': static_cond_dim, 'use_fidelity_enhancer':use_fidelity_enhancer, 'feature_extractor_type':feature_extractor_type})

    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = 'cpu'
    else:
        # device = gpu_device_ind
        device = torch.device('cuda:0')

    # conditional sampling
    # print('evaluating...')
    evaluation = Evaluation(saved_models_dir, dataset_name, static_cond_dim, in_channels, input_length, device, config,
                            use_fidelity_enhancer=use_fidelity_enhancer,
                            feature_extractor_type=feature_extractor_type,
                            use_custom_dataset=use_custom_dataset,
                            kind = "generation").to(device)
    (_, _, xhat), xhat_R = evaluation.sample(static_conditions.shape[0], static_conditions) # (b, c, l)
    x_new = np.transpose(xhat, (0, 2, 1)) # (b, l, c)

    #### To do - Jiayu, add code to convert the generated data into ds.BaseDataFrame
    # will not implement here
    ####


    wandb.finish()

    return x_new    



def generate_embeddings(config: dict,
             saved_models_dir: str,
             dataset_name: str,
             static_cond_dim: int,
             ts_data,
             gpu_device_ind,
             use_fidelity_enhancer:bool,
             feature_extractor_type:str,
             use_custom_dataset:bool=False,
             ):
    in_channels, input_length = config['dataset']['num_features'], config['seq_len']

     # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = 'cpu'
    else:
        # device = gpu_device_ind
        device = torch.device('cuda:0')

    evaluation = Evaluation(saved_models_dir, dataset_name, static_cond_dim, in_channels, input_length, device, config,
                            use_fidelity_enhancer=use_fidelity_enhancer,
                            feature_extractor_type=feature_extractor_type,
                            use_custom_dataset=use_custom_dataset,
                            kind = "embedding").to(device)
    
    z_low_freq, z_high_freq = evaluation.extract_embeddings(ts_data.shape[0], ts_data)

    return z_low_freq, z_high_freq

    


if __name__ == '__main__':
    # load config
    args = load_args()
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    else:
        config = load_yaml_param_settings(args.config)
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['evaluation']['batch_size']
    static_cond_dim = config['static_cond_dim']
    seq_len = config['seq_len']
    gpu_device_ind = config['gpu_device_id']
    saved_models_dir = args.saved_models_dir
    dataset_importer = DatasetImporterCustom(config=config, train_data_path=None,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')
    static_conditions = torch.from_numpy(test_data_loader.dataset.SC)
    # print(static_conditions.shape)
    # generate synthetic data
    generate_data(config, saved_models_dir, dataset_name, static_cond_dim, static_conditions, gpu_device_ind, use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True)

    # clean memory
    torch.cuda.empty_cache()

