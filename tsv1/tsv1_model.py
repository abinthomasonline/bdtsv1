import json
import os
import shutil

import numpy as np
import pandas as pd
import torch

from tsv1.stage1 import *
from tsv1.stage2 import *
from tsv1.generate import *
from tsv1.utils import freeze
from tsv1.experiments.exp_stage1 import ExpStage1
from tsv1.experiments.exp_stage2 import ExpStage2

from datapip import data_struct as ds
from datapip.algo.basic import uniform_like


class ts_v1_model:
    """Implements TS-V1 model class
      Usage:
      1. TS model class for relational model

      :param static_train_data: static training data
      :type static_train_data: ds.BaseDataFrame
      :param temporal_train_data: temporal training data
      :type temporal_train_data: ds.BaseDataFrameGroupBy
      :param num_features: number of features
      :type num_features: int
      :param static_cond_dim: number of static condition dimensions
      :type static_cond_dim: int
      :param dataset_name: name of the dataset  
      :type dataset_name: str
      :param seq_len: length of the sequence
      :type seq_len: int
      :param chunk_size: size of batch for processing
      :type chunk_size: int
      :param out_dir: directory to save the model
      :type out_dir: str
    """
    def __init__(self, static_train_data: ds.BaseDataFrame=None, temporal_train_data: ds.BaseDataFrameGroupBy=None, 
                 dataset_name: str=None, seq_len: int=None, num_features: int=None, static_cond_dim: int=None, chunk_size: int=32, out_dir: str=None, **kwargs):
        # Use __file__ to get the absolute path of the current module, then construct the config path relative to it
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(current_dir, "/configs/model/config.json")
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.num_ts_features = num_features
        self.static_cond_dim = static_cond_dim
        self.static_train_data = static_train_data
        self.temporal_train_data = temporal_train_data
        self.chunk_size = chunk_size
        self.out_dir = out_dir

        #### To do - Jiayu, add code to split static_train_data and temporal_train_data into static_test_data and temporal_test_data for validation in early stopping
        self.static_test_data = None
        self.temporal_test_data = None

        ####
        
    
    def train(self):
        """
        Train the TSV1 model
        """
        is_test = ds.BaseSeries.from_uniform(True, index=self.static_train_data.index)
        is_test = uniform_like(is_test, low=0, high=1) <= 0.2
        static_test_data = self.static_train_data[is_test]
        temporal_test_data = self.temporal_train_data.filter(lambda g, d: is_test[g])

        new_model_config_save_path = self.config_path

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
        
        train_stage1(config, self.out_dir, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)
        model_config['data']['dataset'] = config['dataset']
        os.remove(new_model_config_save_path)
        with open(new_model_config_save_path, "w") as f:
            json.dump(model_config, f, indent=4)

        
        # load training configs for Stage2
        config = self.load_config()

        # Stage 2 training
        dataset_name = self.dataset_name
        batch_size = config['dataset']['batch_sizes']['stage2']
        static_cond_dim = self.static_cond_dim
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        if static_cond_dim != config['static_cond_dim'] or static_cond_dim != len(self.static_train_data.columns):
            raise ValueError(f"static_cond_dim mismatch: {static_cond_dim} != {config['static_cond_dim']} or {len(self.static_train_data.columns)}")

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=self.static_train_data, 
                                                 temporal_data_train=self.temporal_train_data, 
                                                 static_data_test=static_test_data, 
                                                 temporal_data_test=temporal_test_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                               for kind in ['train', 'test']]
        
        train_stage2(config, self.out_dir, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
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
                                                 static_data_test=self.static_test_data, 
                                                 temporal_data_test=self.temporal_test_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)
        
        train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                               for kind in ['train', 'test']]
        
        train_stage1(config, self.out_dir, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)

        

    def generate_ts_data(self, static_condition_data: ds.BaseDataFrame):
        config = self.load_config()
        dataset_name = self.dataset_name
        batch_size = config['evaluation']['batch_size']
        static_cond_dim = self.static_cond_dim
        seq_len = self.seq_len
        gpu_device_ind = config['gpu_device_id']

        if static_cond_dim != config['static_cond_dim'] or static_cond_dim != len(static_condition_data.columns):
            raise ValueError(f"static_cond_dim mismatch: {static_cond_dim} != {config['static_cond_dim']} or {len(static_condition_data.columns)}")

        dataset_importer = DatasetImporterCustom(config=config, static_data_train=None,
                                                 temporal_data_train=None,
                                                 static_data_test=static_condition_data,
                                                 temporal_data_test=None,
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)

        test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')

        index = test_data_loader.dataset.SC.index
        outputs = []
        if self.temporal_train_data.columns.nlevels > 1:
            index_column = tuple(["$index"] + [""] * (self.temporal_train_data.columns.nlevels - 1))
            all_columns = [index_column, *self.temporal_train_data.columns]
            all_columns = ds.BaseIndex.registry[static_condition_data.data_struct].from_tuples(all_columns)
        else:
            all_columns = ["$index", *self.temporal_train_data.columns]
            index_column = "$index"

        for st in range(0, len(index), self.chunk_size):
            static_conditions = torch.from_numpy(
                test_data_loader.dataset.SC.get_by_index(index[st:st + self.chunk_size]).values
            )

            # generate synthetic data
            syn_data = generate_data(
                config, self.out_dir, dataset_name, static_cond_dim, static_conditions, gpu_device_ind,
                use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True
            )

            # clean memory
            torch.cuda.empty_cache()
            ####
            syn_data = syn_data.view(syn_data.shape[0], self.seq_len, -1)
            syn_data = torch.cat([
                torch.arange(syn_data.shape[0]).view(-1, 1, 1).repeat(1, syn_data.shape[1], 1) + st, syn_data
            ], dim=-1).view(-1, syn_data.shape[-1] + 1)
            outputs.append(static_condition_data.from_pandas(
                pd.DataFrame(syn_data.numpy(), columns=all_columns)
            ))

        outputs = static_condition_data.concat(outputs, ignore_index=True, axis=0)

        return outputs.groupby(index_column)[self.temporal_train_data.columns]



    def generate_embeddings(self, ts_data: ds.BaseDataFrameGroupBy):
        config = self.load_config()
        dataset_name = config['dataset']['dataset_name']
        batch_size = config['evaluation']['batch_size']
        static_cond_dim = self.static_cond_dim
        seq_len = config['seq_len']
        gpu_device_ind = config['gpu_device_id']


        index = ts_data.groups
        dataset_importer = DatasetImporterCustom(config=config, static_data_train=None,
                                                 temporal_data_train=None, 
                                                 static_data_test=ds.BaseSeries.registry[ts_data.data_struct].from_uniform(0, index=index)[[]],
                                                 temporal_data_test=ts_data, 
                                                 seq_len=seq_len, data_scaling=True, batch_size=self.chunk_size)

        test_data_loader = build_custom_data_pipeline(batch_size, dataset_importer, config, 'test')
        low_outputs = []
        high_outputs = []
        for st in range(0, len(index), self.chunk_size):
            group_index = index[st:st + self.chunk_size]
            ts_data = torch.from_numpy(test_data_loader.dataset.TS.get_by_index(group_index).values)

            # generate embeddings
            low_freq_embeddings, high_freq_embeddings = generate_embeddings(
                config, self.out_dir, dataset_name, static_cond_dim, ts_data, gpu_device_ind,
                use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True
            )

            # clean memory
            torch.cuda.empty_cache()
            ####
            low_freq_embeddings = ds.BaseDataFrame.registry[index.data_struct].from_pandas(pd.DataFrame(low_freq_embeddings), index=group_index)
            high_freq_embeddings =  ds.BaseDataFrame.registry[index.data_struct].from_pandas(pd.DataFrame(high_freq_embeddings), index=group_index)
            low_outputs.append(low_freq_embeddings)
            high_outputs.append(high_freq_embeddings)

        return (ds.BaseDataFrame.registry[index.data_struct].concat(low_outputs, axis=0),
                ds.BaseDataFrame.registry[index.data_struct].concat(high_outputs, axis=0))


    # def save(self):
    #     os.makedirs(self.out_dir, exist_ok=True)
    #     dataset_name = self.dataset_name
    #     model_save_path = os.path.join(f'saved_models', f'stage1-{dataset_name}.ckpt')
    #     if os.path.exists(model_save_path): 
    #         shutil.copy(model_save_path, self.out_dir)
    #         print(f'Stage 1 model saved to {self.out_dir}')
    #     else:
    #         print(f'Stage 1 model not found at {model_save_path}')
    #     model_save_path = os.path.join(f'saved_models', f'stage2-{dataset_name}.ckpt')
    #     if os.path.exists(model_save_path):
    #         shutil.copy(model_save_path, self.out_dir)
    #         print(f'Stage 2 model saved to {self.out_dir}')
    #     else:
    #         print(f'Stage 2 model not found at {model_save_path}')
    
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
        config['static_cond_dim'] = self.static_cond_dim

        return config
    
    def load_model_stage1(self, model_path: str):
        model_stage1 = ExpStage1.load_from_checkpoint(model_path, in_channels=self.num_ts_features, input_length=self.seq_len, 
                                               config=self.load_config(), map_location='cpu')
        freeze(model_stage1)
        model_stage1.eval()

        # stage 1 model has the following components:
        # encoder_l = model_stage1.encoder_l
        # decoder_l = model_stage1.decoder_l
        # vq_model_l = model_stage1.vq_model_l
        # encoder_h = model_stage1.encoder_h
        # decoder_h = model_stage1.decoder_h
        # vq_model_h = model_stage1.vq_model_h
        return model_stage1

    def load_model_stage2(self, model_path: str):
        model_stage2 = ExpStage2.load_from_checkpoint(model_path, dataset_name=self.dataset_name,
                                               static_cond_dim=self.static_cond_dim,
                                               in_channels=self.num_ts_features,
                                               input_length=self.seq_len, 
                                               config=self.load_config(),
                                               use_fidelity_enhancer=False,
                                               feature_extractor_type='rocket',
                                               use_custom_dataset=True,
                                               map_location='cpu',
                                               strict=False)
        freeze(model_stage2)
        model_stage2.eval()
        return model_stage2.maskgit
    

    def print_config_path(self):
        print(self.config_path)
    
