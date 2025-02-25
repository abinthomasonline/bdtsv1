"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from datapip import data_struct as ds

# Change this import to absolute import
from tsv1.utils import get_root_dir




class DatasetImporterCustom(object):
    def __init__(self, config, static_data_train: ds.BaseDataFrame = None, temporal_data_train: ds.BaseDataFrameGroupBy = None, static_data_test: ds.BaseDataFrame = None, temporal_data_test: ds.BaseDataFrameGroupBy = None, seq_len: int = 0, data_scaling: bool = True, batch_size: int = 32, **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mean = None
        self.std = None

        # Process training data in batches
        if static_data_train is not None and temporal_data_train is not None:
            # First pass: calculate mean and std
            if data_scaling:
                self.calculate_statistics(temporal_data_train)
                config['dataset']['mean'] = self.mean.tolist()
                config['dataset']['std'] = self.std.tolist()
            
            # Second pass: process and scale data
            self.process_data('train', static_data_train, temporal_data_train, data_scaling)
            config['dataset']['num_features'] = self.TS_train.shape[1]
            print(f"TS_train shape: {self.TS_train.shape}")
            print(f"SC_train shape: {self.SC_train.shape}")

        # Process test data in batches
        if static_data_test is not None:
            self.process_data('test', static_data_test, temporal_data_test, data_scaling)

    # To do - Jiayu, modify this function to use datapip
    def calculate_statistics(self, temporal_data: ds.BaseDataFrameGroupBy):
        """Calculate mean and std in batches"""
        sum_x = 0
        sum_x2 = 0
        total_count = 0
        
        # First pass: calculate mean
        chunks = pd.read_csv(data_path, skiprows=1, header=None, chunksize=self.batch_size)
        for chunk in chunks:
            chunk = chunk.astype('float32')
            ts = chunk.iloc[:, self.static_cond_dim:].values
            if ts.shape[1] // self.seq_len != 0:
                raise ValueError("The number of time series in the dataset is not divisible by the sequence length.")
            else:
                ts = np.reshape(ts, (ts.shape[0], ts.shape[1] // self.seq_len, self.seq_len)) # (b, c, l)
                
                sum_x += np.nansum(ts, axis=(0, 2))
                sum_x2 += np.nansum(ts ** 2, axis=(0, 2))
                total_count += ts.shape[0] * ts.shape[2]  # batch_size * seq_len
        
        # Calculate mean and std
        self.mean = (sum_x / total_count)[None, :, None]  # (1 c 1)
        variance = (sum_x2 / total_count) - (self.mean.squeeze() ** 2)
        self.std = np.sqrt(variance)[None, :, None]  # (1 c 1)
        
        # Handle zero std
        self.std[self.std == 0] = 1.0

        self.mean.astype(np.float32)
        self.std.astype(np.float32)

    def process_data(self, kind: str, static_data: ds.BaseDataFrame, temporal_data: ds.BaseDataFrameGroupBy, data_scaling: bool):
        ts_list = []
        sc_list = []

        static_ids = static_data.index
        if temporal_data is not None:
            n_temporal_columns = len(temporal_data.columns)
        else:
            n_temporal_columns = 0

        for i in range(0, static_data.shape[0], self.batch_size):
            # Split into time series and static conditions
            batch_ids = static_ids[i:i + self.batch_size]
            sc = static_data[batch_ids].values
            if n_temporal_columns > 0:
                ts = np.zeros((sc.shape[0], self.seq_len * n_temporal_columns), dtype=sc.dtype)
                for j, sc_id in enumerate(batch_ids):
                    g_data = temporal_data.get_group(sc_id).values.flatten()
                    ts[j, :g_data.shape[-1]] = g_data
            else:
                ts = np.zeros((sc.shape[0], 0), dtype=sc.dtype)

            # Ensure chunk size is divisible by sequence length
            if ts.shape[1] // self.seq_len != 0:
                raise ValueError("The number of time series in the dataset is not divisible by the sequence length.")
            else:
                # Process time series data
                ts = np.reshape(ts, (ts.shape[0], ts.shape[1] // self.seq_len, self.seq_len)) # (b, c, l)
                    
                # Scale the batch if needed
                if data_scaling and self.mean is not None and self.std is not None:
                    ts = (ts - self.mean) / self.std
                
                
                # Handle NaN values for the current batch
                np.nan_to_num(ts, copy=False)
                
                ts_list.append(static_data.from_pandas(pd.DataFrame(ts), index=batch_ids))
                sc_list.append(static_data.from_pandas(pd.DataFrame(sc), index=batch_ids))
        
        # Concatenate all processed chunks
        if ts_list:
            if kind == 'train':
                self.TS_train = static_data.concat(ts_list, axis=0)
                self.SC_train = static_data.concat(sc_list, axis=0)
            else:
                self.TS_test = static_data.concat(ts_list, axis=0)
                self.SC_test = static_data.concat(sc_list, axis=0)

class CustomDataset(Dataset):
    def __init__(self, kind: str, dataset_importer: DatasetImporterCustom, **kwargs):
        """
        :param kind: "train" | "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['train', 'test']
        self.kind = kind

        if kind == "train":
            self.TS, self.SC = dataset_importer.TS_train, dataset_importer.SC_train
        elif kind == "test":
            self.TS, self.SC = dataset_importer.TS_test, dataset_importer.SC_test
        else:
            raise ValueError

        self._len = self.TS.shape[0]
        self._index = self.TS.index

    def __getitem__(self, idx):
        ts, sc = self.TS.get_by_index(self._index[idx]), self.SC.get_by_index(self._index[idx])
        return ts, sc

    def __len__(self):
        return self._len