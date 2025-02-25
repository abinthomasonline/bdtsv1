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

def sliding_window_view(data, window_size, step=1):
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L // window_size
    # B = L - window_size + 1

    # Shape of the output array
    new_shape = (B, window_size, C)

    # Calculate strides
    original_strides = data.strides
    new_strides = (window_size * original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)
    # new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    # strided_array = np.transpose(strided_array, axes=(0, 2, 1)) #(b c l)
    return strided_array


class DatasetImporterCustom(object):
    def __init__(self, config, train_data_path: str, test_data_path: str, static_cond_dim: int, seq_len: int, data_scaling: bool = True, batch_size: int = 32, **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.static_cond_dim = static_cond_dim
        self.mean = None
        self.std = None

        # Process training data in batches
        if train_data_path:
            # First pass: calculate mean and std
            if data_scaling:
                self.calculate_statistics(train_data_path)
                config['dataset']['mean'] = float(self.mean)
                config['dataset']['std'] = float(self.std)
            
            # Second pass: process and scale data
            self.process_data('train', train_data_path, config, data_scaling)

        # Process test data in batches
        if test_data_path:
            self.process_data('test', test_data_path, config, data_scaling)

    def calculate_statistics(self, data_path: str):
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
                ts = sliding_window_view(ts, window_size=self.seq_len)
                ts = np.transpose(ts, axes=(0, 2, 1))  # (b c l)
                
                sum_x += np.nansum(ts, axis=(0, 2))
                sum_x2 += np.nansum(ts ** 2, axis=(0, 2))
                total_count += ts.shape[0] * ts.shape[2]  # batch_size * seq_len
        
        # Calculate mean and std
        self.mean = (sum_x / total_count)[None, :, None]  # (1 c 1)
        variance = (sum_x2 / total_count) - (self.mean.squeeze() ** 2)
        self.std = np.sqrt(variance)[None, :, None]  # (1 c 1)
        
        # Handle zero std
        self.std[self.std == 0] = 1.0

    def process_data(
            self, kind: str, static_data: ds.BaseDataFrame, temporal_data: ds.BaseDataFrameGroupBy,
            config: dict, data_scaling: bool
    ):
        ts_list = []
        sc_list = []

        static_ids = static_data.index
        n_temporal_columns = len(temporal_data.columns)
        for i in range(0, static_data.shape[0], self.batch_size):
            # Split into time series and static conditions
            batch_ids = static_ids[i:i + self.batch_size]
            sc = static_data[batch_ids].values
            ts = np.zeros((sc.shape[0], self.seq_len * n_temporal_columns), dtype=sc.dtype)
            for j, sc_id in enumerate(batch_ids):
                g_data = temporal_data.get_group(sc_id).values.flatten()
                ts[j, :g_data.shape[-1]] = g_data

            # Ensure chunk size is divisible by sequence length
            if ts.shape[-1] // self.seq_len != 0:
                raise ValueError("The number of time series in the dataset is not divisible by the sequence length.")
            else:
                # Process time series data
                ts = sliding_window_view(ts, window_size=self.seq_len)
                ts = np.transpose(ts, axes=(0, 2, 1))  # (b c l)
                
                # Scale the batch if needed
                if data_scaling and self.mean is not None and self.std is not None:
                    ts = (ts - self.mean) / self.std
                
                # Process static conditions
                sc = sliding_window_view(sc, window_size=self.seq_len)
                sc = np.delete(sc, obj=np.s_[1:], axis=1)
                sc = np.squeeze(sc)
                
                # Handle NaN values for the current batch
                np.nan_to_num(ts, copy=False)
                
                ts_list.append(static_data.from_pandas(pd.DataFrame(ts), index=batch_ids))
                sc_list.append(static_data.from_pandas(pd.DataFrame(sc), index=batch_ids))
        
        # Concatenate all processed chunks
        if ts_list:
            if kind == 'train':
                self.TS_train = static_data.concat(ts_list, axis=0)
                self.SC_train = static_data.concat(sc_list, axis=0)
                config['dataset']['num_features'] = self.TS_train.shape[1]
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