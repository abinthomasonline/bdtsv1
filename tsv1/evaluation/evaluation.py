"""
FID, IS, JS divergence.
"""
import os
from typing import List, Union, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..experiments.exp_stage2 import ExpStage2
from ..generators.maskgit import MaskGIT
from ..preprocessing.data_pipeline import build_custom_data_pipeline
from ..preprocessing.preprocess import DatasetImporterCustom
from ..generators.sample import static_condition_sample, extract_embedding_for_relational_components
from supervised_FCN_2.example_pretrained_model_loading import load_pretrained_FCN
from supervised_FCN_2.example_compute_FID import calculate_fid
from supervised_FCN_2.example_compute_IS import calculate_inception_score
from ..utils import time_to_timefreq, timefreq_to_time
from ..generators.fidelity_enhancer import FidelityEnhancer
from ..evaluation.rocket_functions import generate_kernels, apply_kernels
from ..utils import zero_pad_low_freq, zero_pad_high_freq, remove_outliers
from ..evaluation.stat_metrics import marginal_distribution_difference, auto_correlation_difference, skewness_difference, kurtosis_difference
from ..experiments.exp_stage1 import ExpStage1
from ..utils import freeze


class Evaluation(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """
    def __init__(self, 
                 saved_models_dir: str,
                 dataset_name: str,
                 static_cond_dim: int,
                 in_channels:int,
                 input_length:int,
                 device:int, 
                 config:dict, 
                 use_fidelity_enhancer:bool=False,
                 feature_extractor_type:str='rocket',
                 rocket_num_kernels:int=1000,
                 use_custom_dataset:bool=True,
                 kind:str=None
                 ):
        super().__init__()
        self.saved_models_dir = saved_models_dir
        self.dataset_name = dataset_name
        self.device = torch.device(device)
        self.config = config
        self.batch_size = self.config['evaluation']['batch_size']
        self.feature_extractor_type = feature_extractor_type
        assert feature_extractor_type in ['supervised_fcn', 'rocket'], 'unavailable feature extractor type.'

        if feature_extractor_type == 'rocket':
            self.rocket_kernels = generate_kernels(input_length, num_kernels=rocket_num_kernels)

        self.mean = self.config['dataset']['mean'] # scaling coefficient
        self.std = self.config['dataset']['std']  # scaling coefficient

        self.ts_len = self.config['seq_len']  # time series length (seq_len)

        # load the stage2 model
        if kind == "generation":
            self.stage2 = ExpStage2.load_from_checkpoint(os.path.join(self.saved_models_dir, f'stage2.ckpt'), 
                                                      saved_models_dir=self.saved_models_dir,
                                                      dataset_name=dataset_name,
                                                      static_cond_dim=static_cond_dim,
                                                      in_channels=in_channels,
                                                      input_length=input_length, 
                                                      config=config,
                                                      use_fidelity_enhancer=False,
                                                      feature_extractor_type=feature_extractor_type,
                                                      use_custom_dataset=use_custom_dataset,
                                                      map_location='cpu',
                                                      strict=False)
            self.stage2.eval()
            self.maskgit = self.stage2.maskgit
            self.stage1 = self.stage2.maskgit.stage1
        elif kind == "embedding":
            num_ts_features = self.config['dataset']['num_features']
            self.stage1 = ExpStage1.load_from_checkpoint(os.path.join(self.saved_models_dir, f'stage1.ckpt'), 
                                               in_channels=num_ts_features, input_length=self.ts_len, 
                                               config=self.config, map_location='cpu')
            freeze(self.stage1)
            self.stage1.eval()

        # load the fidelity enhancer
        if use_fidelity_enhancer:
            self.fidelity_enhancer = FidelityEnhancer(self.ts_len, 1, config)
            fname = f'fidelity_enhancer .ckpt'
            ckpt_fname = os.path.join('saved_models', fname)
            self.fidelity_enhancer.load_state_dict(torch.load(ckpt_fname))
        else:
            self.fidelity_enhancer = nn.Identity()


    @torch.no_grad()
    def sample(self, n_samples: int, static_condition, unscale:bool=True):
        """
        unscale: unscale the generated sample with percomputed mean and std.
        """

        #sampling
        x_new_l, x_new_h, x_new = static_condition_sample(self.maskgit, n_samples, self.device, static_condition, self.batch_size)  # (b c l); b=n_samples, c=1 (univariate)


        # FE
        num_batches = x_new.shape[0] // self.batch_size + (1 if x_new.shape[0] % self.batch_size != 0 else 0)
        X_new_R = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            mini_batch = x_new[start_idx:end_idx]
            x_new_R = self.fidelity_enhancer(mini_batch.to(self.device)).cpu()
            X_new_R.append(x_new_R)
        X_new_R = torch.cat(X_new_R)

        # unscale
        if unscale:
            # Convert mean and std to tensors with proper shape if they're not already
            if not isinstance(self.mean, torch.Tensor):
                mean = torch.tensor(self.mean, dtype=torch.float32)
            else:
                mean = self.mean
                
            if not isinstance(self.std, torch.Tensor):
                std = torch.tensor(self.std, dtype=torch.float32)
            else:
                std = self.std
                
            # Reshape mean and std if needed
            if isinstance(mean, torch.Tensor) and mean.dim() == 0:
                mean = mean.item()  # Convert single element tensor to scalar
            if isinstance(std, torch.Tensor) and std.dim() == 0:
                std = std.item()  # Convert single element tensor to scalar
                
            # Apply unscaling
            x_new_l = x_new_l * std + mean
            x_new_h = x_new_h * std + mean
            x_new = x_new * std + mean
            X_new_R = X_new_R * std + mean

        return (x_new_l, x_new_h, x_new), X_new_R
    

    @torch.no_grad()
    def extract_embeddings(self, n_samples, x):
        """
        extract embeddings from the encoder
        """
        z_low_freq, z_high_freq = extract_embedding_for_relational_components(self.stage1, n_samples, self.device, x, self.batch_size)
        return z_low_freq, z_high_freq
