import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
import datetime
from datetime import timedelta, date
import json


def source_to_data_pipeline(source_data_path, seq_len, num_non_ts_cols):
    df = pd.read_csv(source_data_path)
    row_count, col_count = df.shape
    num_ts_cols = col_count - num_non_ts_cols

    print("number of non ts columns: {}, number of ts columns: {}".format(num_non_ts_cols, num_ts_cols))
    ts = df[df.columns[num_non_ts_cols:]]
    ts_col_names = list(ts.columns)
    extracted_ts_col_names = []
    if len(ts_col_names) != num_ts_cols:
        raise Exception("Number of ts columns names does not match number ts columns")
    for i in range(num_ts_cols):
        if i % seq_len == 0:
            extracted_ts_col_names.append(ts_col_names[i])

    ts = ts.to_numpy()
    print("temporal data shape: {}, seq_len is: {}".format(ts.shape, seq_len))
    if ts.shape[1] % seq_len != 0:
        raise Exception("length of time series data must be divisble by seq_len")
    ts = np.reshape(ts, (ts.shape[0], seq_len, ts.shape[1] // seq_len))
    dim = ts.shape[0] * ts.shape[1]
    dim2 = ts.shape[2]
    ts = np.reshape(ts, (dim, dim2))
    # convert ts data back to df
    ts_df = pd.DataFrame(ts, columns=extracted_ts_col_names)

    # extract static data and repeat seq_len times
    static_df = df[df.columns[:num_non_ts_cols]]
    static_df_new = pd.DataFrame(np.repeat(static_df.values, seq_len, axis=0))
    static_df_new.columns = static_df.columns
    # merge two dfs
    df_to_data_pipeline = pd.concat([static_df_new, ts_df], axis=1)

    return df_to_data_pipeline

