import json
import logging
import os
import signal
import sys
import traceback
import warnings
import shutil

from tsv1.stage1 import *
from tsv1.stage2 import *
from tsv1.generate import *
from tsv1.utils import get_root_dir

from datapip import data_struct as ds

warnings.filterwarnings('ignore')


class tsv1:
    """Implements TS-V1 model class
      Usage:
      1. TS model class for relational model

      :param static_train_data: static training data
      :type static_train_data: ds.BaseDataFrame
      :param temporal_train_data: temporal training data
      :type temporal_train_data: ds.BaseDataFrameGroupBy
      :param static_condition_data: static condition data for generation
      :type static_condition_data: ds.BaseDataFrame
      :param config_path: path to configuration file
      :type config_path: str
      :param chunk_size: size of batch for processing
      :type chunk_size: int
      :param out_dir: directory to save the model
      :type out_dir: str
    """
    def __init__(self, static_train_data: ds.BaseDataFrame=None, temporal_train_data: ds.BaseDataFrameGroupBy=None, 
                 static_condition_data: ds.BaseDataFrame=None, dataset_name: str=None, seq_len: int=1, chunk_size: int=32, out_dir: str=None, **kwargs):
        self.config_path = "../config/config.json"
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.static_train_data = static_train_data
        self.temporal_train_data = temporal_train_data
        self.static_condition_data = static_condition_data
        self.chunk_size = chunk_size
        self.out_dir = out_dir
    
    def train(self):
        """
        Train the TSV1 model
        """
        #### To do - Jiayu, add code to split static_train_data and temporal_train_data into static_test_data and temporal_test_data for validation in early stopping


        ####
        with open(self.config_path, "r") as f:
            model_config = json.load(f)
            f.close()

        new_model_config_save_path = self.config_path
        config = self.load_config()

        # Stage1 training
        dataset_name = self.dataset_name
        batch_size = config['dataset']['batch_sizes']['stage1']
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=self.static_train_data, 
                                                 temporal_data_train=self.temporal_train_data, 
                                                 static_data_test=static_test_data, 
                                                 temporal_data_test=temporal_test_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                               for kind in ['train', 'test']]
        
        train_stage1(config, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)
        model_config['data']['dataset'] = config['dataset']
        os.remove(new_model_config_save_path)
        with open(new_model_config_save_path, "w") as f:
            json.dump(model_config, f)

        
        # load training configs for Stage2
        config = self.load_config()

        # Stage 2 training
        dataset_name = self.dataset_name
        batch_size = config['dataset']['batch_sizes']['stage2']
        static_cond_dim = len(self.static_train_data.columns)
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=self.static_train_data, 
                                                 temporal_data_train=self.temporal_train_data, 
                                                 static_data_test=static_test_data, 
                                                 temporal_data_test=temporal_test_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                               for kind in ['train', 'test']]
        
        train_stage2(config, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
                 feature_extractor_type='rocket', use_custom_dataset=True)
        

    def train_stage1(self):
        """
        Train the TSV1 stage 1 model
        """
        config = self.load_config()
        # Stage1 training
        dataset_name = self.dataset_name
        batch_size = config['dataset']['batch_sizes']['stage1']
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=self.static_train_data, 
                                                 temporal_data_train=self.temporal_train_data, 
                                                 static_data_test=static_test_data, 
                                                 temporal_data_test=temporal_test_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                               for kind in ['train', 'test']]
        
        train_stage1(config, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)

        

    def generate_data(self):
        config = self.load_config()
        dataset_name = self.dataset_name
        batch_size = config['evaluation']['batch_size']
        static_cond_dim = config['static_cond_dim']
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=None, 
                                                 temporal_data_train=None, 
                                                 static_data_test=self.static_condition_data, 
                                                 temporal_data_test=None, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')

        # To do - Jiayu, check if this can be loaded properly
        static_conditions = torch.from_numpy(test_data_loader.dataset.SC)

        # generate synthetic data
        syn_data = generate_data(config, dataset_name, static_cond_dim, static_conditions, gpu_device_ind, use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True)

        # clean memory
        torch.cuda.empty_cache()

        return syn_data



    def generate_embeddings(self):
        config = self.load_config()
        dataset_name = config['dataset']['dataset_name']
        batch_size = config['evaluation']['batch_size']
        static_cond_dim = config['static_cond_dim']
        seq_len = config['seq_len']
        gpu_device_ind = config['gpu_device_id']

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=None, 
                                                 temporal_data_train=None, 
                                                 static_data_test=self.static_condition_data, 
                                                 temporal_data_test=None, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')
        
        # To do - Jiayu, check if this can be loaded properly
        ts_data = torch.from_numpy(test_data_loader.dataset.TS)

        # generate embeddings
        low_freq_embeddings, high_freq_embeddings = generate_embeddings(config, dataset_name, static_cond_dim, ts_data, gpu_device_ind, use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True)

        # clean memory
        torch.cuda.empty_cache()

        return low_freq_embeddings, high_freq_embeddings


    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        dataset_name = self.dataset_name
        model_save_path = os.path.join(f'saved_models', f'stage1-{dataset_name}.ckpt')
        if os.path.exists(model_save_path): 
            shutil.copy(model_save_path, self.out_dir)
            print(f'Stage 1 model saved to {self.out_dir}')
        else:
            print(f'Stage 1 model not found at {model_save_path}')
        model_save_path = os.path.join(f'saved_models', f'stage2-{dataset_name}.ckpt')
        if os.path.exists(model_save_path):
            shutil.copy(model_save_path, self.out_dir)
            print(f'Stage 2 model saved to {self.out_dir}')
        else:
            print(f'Stage 2 model not found at {model_save_path}')
    
    def load_config(self):
        with open(self.config_path, "r") as f:
            model_config = json.load(f)
            f.close()

        if 'seq_len' in model_config:
            config = model_config
        else:
            config = {}
            for d in (model_config['data'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)
        
        config['dataset']['dataset_name'] = self.dataset_name
        config['seq_len'] = self.seq_len

        return config


