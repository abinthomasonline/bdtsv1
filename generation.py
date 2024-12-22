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


import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing.data_pipeline import build_custom_data_pipeline
from preprocessing.preprocess import DatasetImporterCustom
import pandas as pd

from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, str2bool


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
    return parser.parse_args()


def evaluate(config: dict,
             dataset_name: str,
             static_cond_dim: int,
             dataset_importer: DatasetImporterCustom,
             static_conditions,
             train_data_loader: DataLoader,
             gpu_device_idx,
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

    # n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.TS.shape
    
    # wandb init
    wandb.init(project='TimeVQVAE-evaluation', 
               config={**config, 'dataset_name': dataset_name, 'static_cond_dim': static_cond_dim, 'use_fidelity_enhancer':use_fidelity_enhancer, 'feature_extractor_type':feature_extractor_type})

    # unconditional sampling
    print('evaluating...')
    evaluation = Evaluation(dataset_name, static_cond_dim, dataset_importer, in_channels, input_length, gpu_device_idx, config,
                            use_fidelity_enhancer=use_fidelity_enhancer,
                            feature_extractor_type=feature_extractor_type,
                            use_custom_dataset=use_custom_dataset).to(gpu_device_idx)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    # Need to add sample function for static conditional sampling
    (_, _, xhat), xhat_R = evaluation.sample(min(static_conditions.shape[0],min_num_gen_samples), static_conditions)
    x_new = np.transpose(xhat, (0, 2, 1))
    if not os.path.isdir(get_root_dir().joinpath('synthetic_data')):
        os.mkdir(get_root_dir().joinpath('synthetic_data'))
    np_file_path = os.path.join(f'synthetic_data', f'synthetic-{dataset_name}.npy')
    csv_file_path = os.path.join(f'synthetic_data', f'synthetic-{dataset_name}.csv')
    np.save(np_file_path, x_new)
    dim1 = x_new.shape[0] * x_new.shape[1]
    dim2 = x_new.shape[2]
    df = pd.DataFrame(x_new.reshape((dim1, dim2)))
    df.to_csv(csv_file_path)

    # (_, _, xhat), xhat_R = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    # z_train = evaluation.z_train
    # z_test = evaluation.z_test
    # z_rec_train = evaluation.compute_z_rec('train')
    # z_rec_test = evaluation.compute_z_rec('test')
    # zhat = evaluation.compute_z_gen(xhat)

        
    # # class-conditional sampling
    # print('evaluation for class-conditional sampling...')
    # n_plot_samples_per_class = 100 #200
    # alpha = 0.1
    # ylim = (-5, 5)
    # n_rows = int(np.ceil(np.sqrt(n_classes)))
    # fig1, axes1 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    # fig2, axes2 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    # fig3, axes3 = plt.subplots(n_rows, n_rows, figsize=(4*n_rows, 2*n_rows))
    # fig1.suptitle('X_test_c')
    # fig2.suptitle(f"Xhat_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
    # fig3.suptitle(f"Xhat_R_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
    # axes1 = axes1.flatten()
    # axes2 = axes2.flatten()
    # axes3 = axes3.flatten()
    # for cls_idx in range(n_classes):
    #     (_, _, xhat_c), xhat_c_R = evaluation.sample(n_plot_samples_per_class, kind='conditional', class_index=cls_idx)
    #     cls_sample_ind = (evaluation.Y_test[:,0] == cls_idx)  # (b,)
    #
    #     X_test_c = evaluation.X_test[cls_sample_ind]  # (b' 1 l)
    #     sample_ind = np.random.randint(0, X_test_c.shape[0], n_plot_samples_per_class)
    #     axes1[cls_idx].plot(X_test_c[sample_ind,0,:].T, alpha=alpha, color='C0')
    #     axes1[cls_idx].set_title(f'cls_idx:{cls_idx}')
    #     axes1[cls_idx].set_ylim(*ylim)
    #
    #     sample_ind = np.random.randint(0, xhat_c.shape[0], n_plot_samples_per_class)
    #     axes2[cls_idx].plot(xhat_c[sample_ind,0,:].T, alpha=alpha, color='C0')
    #     axes2[cls_idx].set_title(f'cls_idx:{cls_idx}')
    #     axes2[cls_idx].set_ylim(*ylim)
    #
    #     if use_fidelity_enhancer:
    #         sample_ind = np.random.randint(0, xhat_c_R.shape[0], n_plot_samples_per_class)
    #         axes3[cls_idx].plot(xhat_c_R[sample_ind,0,:].T, alpha=alpha, color='C0')
    #         axes3[cls_idx].set_title(f'cls_idx:{cls_idx}')
    #         axes3[cls_idx].set_ylim(*ylim)
    #
    # fig1.tight_layout()
    # fig2.tight_layout()
    # wandb.log({"X_test_c": wandb.Image(fig1)})
    # wandb.log({f"Xhat_c": wandb.Image(fig2)})
    #
    # if use_fidelity_enhancer:
    #     fig3.tight_layout()
    #     wandb.log({f"Xhat_R_c": wandb.Image(fig3)})
    #
    # plt.close(fig1)
    # plt.close(fig2)
    # plt.close(fig3)
    #
    # wandb.finish()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    dataset_name = args.dataset_names[0]
    batch_size = config['evaluation']['batch_size']
    dataset_importer = DatasetImporterCustom(train_data_path=args.train_data_path, test_data_path=args.test_data_path,
                                             static_cond_dim=args.static_cond_dim, seq_len=args.seq_len,
                                             **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for
                                           kind in ['train', 'test']]
    static_conditions = torch.from_numpy(test_data_loader.dataset.SC)
    print(static_conditions.shape)
    # generate synthetic data
    evaluate(config, dataset_name, args.static_cond_dim, dataset_importer, static_conditions, train_data_loader, args.gpu_device_idx,
             args.use_fidelity_enhancer, args.feature_extractor_type, args.use_custom_dataset)

    # clean memory
    torch.cuda.empty_cache()

    # # dataset names
    # if len(args.dataset_names) == 0:
    #     data_summary_ucr = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
    #     dataset_names = data_summary_ucr['Name'].tolist()
    # else:
    #     dataset_names = args.dataset_names
    # print('dataset_names:', dataset_names)
    #
    # for dataset_name in dataset_names:
    #     print('dataset_name:', dataset_name)
    #
    #     # data pipeline
    #     batch_size = config['evaluation']['batch_size']
    #     if not args.use_custom_dataset:
    #         dataset_importer = DatasetImporterUCR(dataset_name, **config['dataset'])
    #         train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
    #     else:
    #         dataset_importer = DatasetImporterCustom(**config['dataset'])
    #         train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
    #
    #     # train
    #     evaluate(config, dataset_name, args.static_cond_dim, train_data_loader, args.gpu_device_idx, args.use_fidelity_enhancer, args.feature_extractor_type, args.use_custom_dataset)
    #
    #     # clean memory
    #     torch.cuda.empty_cache()
    #
