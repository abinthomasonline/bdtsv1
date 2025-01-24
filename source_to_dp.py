import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
import datetime
from datetime import timedelta, date
import json


def source_to_data_pipeline(source_data_path, seq_len, static_id, sortby, other_static_columns):
    df = pd.read_csv(source_data_path)

    # get ts column names
    ts_col_names = [c for c in df.columns if
                    "Unnamed" not in c and c not in other_static_columns and c != static_id and c != sortby]
    extracted_ts_col_names = []
    for i in range(len(ts_col_names)):
        if i % seq_len == 0:
            extracted_ts_col_names.append(ts_col_names[i])
    # extract ts data and convert to the shape required by next step
    ts = df[[c for c in df.columns if c not in other_static_columns and c != static_id and c != sortby]]
    ts = ts.to_numpy()
    # print(extracted_ts_col_names)
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
    static_df = df[[c for c in df.columns if c in other_static_columns or c == static_id or c == sortby]]
    static_df_new = pd.DataFrame(np.repeat(static_df.values, seq_len, axis=0))
    static_df_new.columns = static_df.columns
    # merge two dfs
    df_to_data_pipeline = pd.concat([static_df_new, ts_df], axis=1)

    return df_to_data_pipeline

