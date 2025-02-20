"""
`Dataset` (pytorch) class is defined.
"""
from typing import Union
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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
    def __init__(self, config, train_data_path: str, test_data_path: str, static_cond_dim: int, seq_len: int, data_scaling: bool = True, **kwargs):
        # training and test datasets
        # typically, you'd load the data, for example, using pandas

        # fetch an entire dataset
        # df_train = pd.read_csv(train_data_path, sep='\t', header=None)
        # df_test = pd.read_csv(test_data_path, sep='\t', header=None)
        if train_data_path:
            df_train = pd.read_csv(train_data_path, skiprows=1, header=None)
            df_train = df_train.astype('float32')
            row_count, col_count = df_train.shape
            if row_count % seq_len != 0:
                raise Exception("Data pipeline transformation failed: Number of rows does not divide sequence length")
            self.TS_train = df_train.iloc[:, static_cond_dim:].values # (b n_features/n_channels)
            self.SC_train = df_train.iloc[:, 0:static_cond_dim].values # (b static_cond_dim)
            # Transfer TS_train to (b c l)
            self.TS_train = sliding_window_view(self.TS_train, window_size=seq_len)
            self.TS_train = np.transpose(self.TS_train, axes=(0, 2, 1))
            self.SC_train = sliding_window_view(self.SC_train, window_size=seq_len)
            # remove duplicated condition rows
            self.SC_train = np.delete(self.SC_train, obj=np.s_[1:], axis=1)
            self.SC_train = np.squeeze(self.SC_train)
            config['dataset']['num_features'] = self.TS_train.shape[1]


        if test_data_path:
            df_test = pd.read_csv(test_data_path, skiprows=1, header=None)
            df_test = df_test.astype('float32')
            self.TS_test = df_test.iloc[:, static_cond_dim:].values # (b n_features/n_channels)
            self.SC_test = df_test.iloc[:, 0:static_cond_dim].values # (b static_cond_dim)
            # Transfer TS_test to (b c l)
            self.TS_test = sliding_window_view(self.TS_test, window_size=seq_len)
            self.TS_test = np.transpose(self.TS_test, axes=(0, 2, 1))
            self.SC_test = sliding_window_view(self.SC_test, window_size=seq_len)
            # remove duplicated condition rows
            self.SC_test = np.delete(self.SC_test, obj=np.s_[1:], axis=1)
            self.SC_test = np.squeeze(self.SC_test)

        # self.TS_train, self.TS_test = df_train.iloc[:, static_cond_dim:].values, df_test.iloc[:, static_cond_dim:].values # (b n_features/n_channels)
        # _, n_channels = self.TS_train.shape
        # # print("Number of channels: ", n_channels)
        # self.SC_train, self.SC_test = df_train.iloc[:, 0:static_cond_dim].values, df_test.iloc[:, 0:static_cond_dim].values  # (b static_cond_dim)
        #
        # # Transfer TS_train and TS_test to dimension (b c l)
        # self.TS_train, self.TS_test = sliding_window_view(self.TS_train, window_size=seq_len), sliding_window_view(self.TS_test, window_size=seq_len)
        # self.TS_train, self.TS_test = np.transpose(self.TS_train, axes=(0, 2, 1)), np.transpose(self.TS_test, axes=(0, 2, 1))
        # self.SC_train, self.SC_test = sliding_window_view(self.SC_train, window_size=seq_len), sliding_window_view(self.SC_test, window_size=seq_len)
        # # remove duplicated condition rows
        # self.SC_train, self.SC_test = np.delete(self.SC_train, obj=np.s_[1:], axis=1), np.delete(self.SC_test, obj=np.s_[1:], axis=1)
        # self.SC_train, self.SC_test = np.squeeze(self.SC_train), np.squeeze(self.SC_test)
        # # self.SC_train, self.SC_test = np.repeat(self.SC_train, n_channels, axis=1), np.repeat(self.SC_test, n_channels, axis=1)

        self.mean, self.std = 1., 1.
        if data_scaling:
            if train_data_path:
                self.mean = np.nanmean(self.TS_train, axis=(0, 2))[None, :, None]  # (1 c 1)
                self.std = np.nanstd(self.TS_train, axis=(0, 2))[None, :, None]  # (1 c 1)
                self.TS_train = (self.TS_train - self.mean) / self.std  # (b c l)
            if test_data_path:
                self.TS_test = (self.TS_test - self.mean) / self.std  # (b c l)

        if train_data_path:
            np.nan_to_num(self.TS_train, copy=False)
            config['dataset']['mean'] = float(self.mean)
            config['dataset']['std'] = float(self.std)
        if test_data_path:
            np.nan_to_num(self.TS_test, copy=False)

        # print('self.TS_train.shape:', self.TS_train.shape)
        # print('self.TS_test.shape:', self.TS_test.shape)
        # print('self.SC_train.shape:', self.SC_train.shape)
        # print('self.SC_test.shape:', self.SC_test.shape)

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

    def __getitem__(self, idx):
        ts, sc = self.TS[idx, :], self.SC[idx, :]
        return ts, sc

    def __len__(self):
        return self._len