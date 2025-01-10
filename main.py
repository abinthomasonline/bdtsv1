import argparse
import json
import logging
import os
import signal
import sys
import traceback
import warnings
from datetime import datetime

import pickle
from tabtransformer import TableTransformer, TimeSeriesTransformer

from arfpy import arf
import pandas as pd
import numpy as np

# import functions from original developed TS repo
from stage1 import *
from stage2 import *
from generate import *
from source_to_dp import source_to_data_pipeline



def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", "-l", type=str, default="INFO",
                        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"])
    parser.add_argument("--time", "-t", action="store_true")
    subparsers = parser.add_subparsers(dest="op")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data-path", "-d", required=True)
    prepare_parser.add_argument("--data-policy", "-p", type=str,
                                default="./configs/data/data_config.json")
    prepare_parser.add_argument("--output-path", "-o", default="./out/ts_data_config_learn.json")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data-path", "-d", required=True)
    validate_parser.add_argument("--sample-size", "-s", default=50000)
    validate_parser.add_argument("--data-config", "-p", type=str, default="./out/ts_data_config_learn.json")
    validate_parser.add_argument("--output-path", "-o", default="./out/ts_data_config_learn.json")

    # Define the args for conditions generation using ARF
    arf_parser = subparsers.add_parser("arf")
    arf_parser.add_argument("--data-path", "-d", required=True)
    arf_parser.add_argument("--data-config", "-p", type=str, default="./out/ts_data_config_learn.json")
    arf_parser.add_argument("--model-config", "-m", type=str, default="./configs/model/config.json")
    # arf_parser.add_argument("--sample-size", "-s", default=5000)
    arf_parser.add_argument("--output-path", "-o", default="./datasets/")

    # Define the args for data preprocessing
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--data-path", "-d", required=True)
    preprocess_parser.add_argument("--data-config", "-p", type=str, default="./out/ts_data_config_learn.json")
    preprocess_parser.add_argument("--model-config", "-m", default="./configs/model/config.json")
    preprocess_parser.add_argument("--if_val", "-v", default=False)
    preprocess_parser.add_argument("--if_cond", "-c", default=False)
    preprocess_parser.add_argument("--output-path", "-o", default="./datasets/")


    # Define the args related to model training
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-path", "-d", default="./datasets/train.csv", required=True)
    train_parser.add_argument("--test-data-path", "-v", required=True)
    train_parser.add_argument("--model-config", "-m", type=str, default="./configs/model/config.json")
    # train_parser.add_argument("--static_cond_dim", "-scd", required=True)

    # Define the args related to synthetic data generation (sampling)
    sample_parser = subparsers.add_parser("sample")
    # sample_parser.add_argument("--data-path", "-d", required=True)
    sample_parser.add_argument("--static-cond-path", "-sd", required=True)
    sample_parser.add_argument("--model-config", "-m", type=str, default="./configs/model/config.json")
    sample_parser.add_argument("--output-path", "-o", required=True)
    return parser.parse_args()


def prepare(args: argparse.Namespace):
    learn_args_path = args.data_config
    data = args.data_path
    with open(learn_args_path, "r") as f:
        data_config_args = json.load(f)
    learn_args = data_config_args['ts']
    learn_single_args = data_config_args['single']

    # Define the proper data pipeline for your model
    TimeSeriesTransformer.validate_kwargs(data, learn_args, learning=True)
    TableTransformer.validate_kwargs(data, learn_single_args, learning=True)
    data = load_data(data, **learn_args.get("data_format_args", {}))  # Skip if `data` is `pd.DataFrame` already
    if "data_format_args" in learn_args:
        learn_args.pop("data_format_args")
    if "data_format_args" in learn_single_args:
        learn_single_args.pop("data_format_args")
    data_static_cond = data[[c for c in data.columns if c in other_static_columns]]
    # Learn configs
    transformer_args = TimeSeriesTransformer.learn_args(data, json_compatible=True, **learn_args)
    transformer_single_args = TableTransformer.learn_args(data_static_cond, json_compatible=True, **learn_single_args)
    TimeSeriesTransformer.validate_kwargs(data, transformer_args)
    TableTransformer.validate_kwargs(data_static_cond, transformer_single_args)
    if "data_format_args" in transformer_args:
        transformer_args.pop("data_format_args")
    if "data_format_args" in transformer_single_args:
        transformer_single_args.pop("data_format_args")

    learned_args = {}
    learned_args['ts'] = transformer_args
    learned_args['single'] = transformer_single_args

    directory = os.path.dirname(args.output_path)
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    with open(args.output_path, "w") as f:
        json.dump(learned_args, f, indent=2)


def validate(args: argparse.Namespace):
    data = args.data_path
    with open(args.data_config, "r") as f:
        data_config_args = json.load(f)
    transformer_args = data_config_args['ts']
    transformer_single_args = data_config_args['single']
    data = load_data(data, **learn_args.get("data_format_args", {}))  # Skip if `data` is `pd.DataFrame` already
    if "data_format_args" in learn_args:
        learn_args.pop("data_format_args")
    if "data_format_args" in learn_single_args:
        learn_single_args.pop("data_format_args")
    data_static_cond = data[[c for c in data.columns if c in other_static_columns]]

    TimeSeriesTransformer.validate_kwargs(data, transformer_args)
    TableTransformer.validate_kwargs(data_static_cond, transformer_single_args)


def warmup(args: argparse.Namespace):
    # Conditions generation using arf
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    config = {}
    for d in (model_config['data'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)
    num_generated_samples = config['num_rows']
    ## Data pipeline
    data = args.data_path
    learn_args_path = args.data_config
    with open(learn_args_path, "r") as f:
        data_config_args = json.load(f)
    transformer_single_args = data_config_args['single']
    other_static_columns = transformer_single_args['other_static_columns']
    data = load_data(data, **transformer_single_args.get("data_format_args", {}))
    data_static_cond = data[[c for c in data.columns if c in other_static_columns]]# Skip if `data` is `pd.DataFrame` already
    if "data_format_args" in transformer_single_args:
        transformer_single_args.pop("data_format_args")
    transformer = TableTransformer.make(**transformer_single_args)
    transformer.fit(data_static_cond)
    transformed = transformer.transform(data_static_cond, start_action='drop', end_action='standardize')  # For details of actions, see below
    transformed = transformed.set_axis(["&".join(c) for c in transformed.columns], axis=1)
    ## Train the ARF
    arf_model = arf.arf(x=transformed)
    ## Get density estimates
    arf_model.forde()
    ## Generate data
    synthetic_data = arf_model.forge(n=num_generated_samples)
    ## Inverse the data to source format
    synthetic_data = synthetic_data.set_axis(pd.MultiIndex.from_tuples([tuple(c.split("&")) for c in synthetic_data.columns]), axis=1)
    synthetic_data, _ = transformer.inverse_transform(synthetic_data, start_action='standardize')
    # cond_data_saving_path = args.output_path + "source_condition.csv"
    synthetic_data.to_csv("./datasets/source_condition.csv", index=False)

    # Dataset split (train:val = 9:1)
    source_data = args.data_path
    df = pd.read_csv(source_data)

    # Creating a dataframe with 90% values of original dataframe
    train_set = df.sample(frac=0.9)

    # Creating dataframe with rest of the 10% values
    val_set = df.drop(train_set.index)

    train_set.to_csv("./datasets/source_train.csv", index=False)
    val_set.to_csv("./datasets/source_val.csv", index=False)


    # Dataset transformation from source to meet data pipeline requirements
    source_train_path = "./datasets/source_train.csv"
    source_val_path = "./datasets/source_val.csv"
    source_condition_path = "./datasets/source_condition.csv"
    df_train_to_dp = source_to_data_pipeline(source_train_path, seq_len, static_id, sortby, other_static_columns)
    df_val_to_dp = source_to_data_pipeline(source_val_path, seq_len, static_id, sortby, other_static_columns)
    df_condition_to_dp = source_to_data_pipeline(source_condition_path, seq_len, static_id, sortby, other_static_columns)
    df_train_to_dp.to_csv("./datasets/dp_train.csv", index=False)
    df_val_to_dp.to_csv("./datasets/dp_val.csv", index=False)
    df_condition_to_dp.to_csv("./datasets/dp_condition.csv", index=False)


def preprocess(args: argparse.Namespace):
    # Load configs
    learn_args_path = args.data_config
    data = args.data_path
    with open(learn_args_path, "r") as f:
        data_config_args = json.load(f)
    # Data transformation config for time series
    transformer_args = data_config_args['ts']
    # Data transformation config for single table
    transformer_single_args = data_config_args['single']
    static_id = transformer_single_args['static_id']
    sortby = transformer_single_args['sortby']
    other_static_columns = transformer_single_args['other_static_columns']
    # Time series data pipeline
    data = load_data(data, **transformer_args.get("data_format_args", {}))  # Skip if `data` is `pd.DataFrame` already
    if "data_format_args" in transformer_args:
        learn_args.pop("data_format_args")
    if "data_format_args" in transformer_single_args:
        learn_single_args.pop("data_format_args")
    data_static_cond = data[[c for c in data.columns if c in other_static_columns]]
    transformer = TimeSeriesTransformer.make(**transformer_args)
    transformer.fit(data)
    transformer_single = TableTransformer.make(**transformer_single_args)
    transformer_single.fit(data_static_cond)
    static_cond_dim = transformer_single.tensorized_dim
    augmented = transformer.transform(data, end_action="augment")
    transformed_single = transformer_single.transform(data_static_cond.drop_duplicates(), start_action='drop',
                                                      end_action='tensorize')
    temporal_df = augmented.temporal.obj.copy()
    temporal_df["betterdata_g_index"] = [x for x, in augmented.temporal.group_names]
    static = transformed_single.copy()
    static.columns = ["&".join(c) for c in static.columns]
    static["betterdata_g_index"] = data[static_id].astype(str)
    merged = temporal_df.merge(static, on="betterdata_g_index", how="inner")
    print(temporal_df.columns.tolist(), other_static_columns)
    temporal_columns = [c for c in data.columns if c not in other_static_columns and c != static_id and c != sortby]
    temporal_columns = [c for c in temporal_df.columns if any(c.startswith(cc + " |||") for cc in temporal_columns)]
    df_final = merged[static.columns.tolist() + temporal_columns].drop(columns=["betterdata_g_index"])

    # save transformed data
    if args.if_val:
        data_saving_path = args.output_path + 'val.csv'
    elif args.if_cond:
        data_saving_path = args.output_path + 'condition.csv'
    else:
        data_saving_path = args.output_path + 'train.csv'
    df_final.to_csv(data_saving_path, index=False)

    # update static_cond_dim in model_config.json
    new_model_config_save_path = args.model_config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
        f.close()

    model_config['train']['static_cond_dim'] = static_cond_dim
    os.rename(new_model_config_save_path)

    with open(new_model_config_save_path, "w") as f:
        json.dump(model_config, f)




def train(args: argparse.Namespace):
    # load training configs for Stage1
    new_model_config_save_path = args.model_config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
        f.close()
    config = {}
    for d in (model_config['data'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)

    # Stage1 training
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['dataset']['batch_sizes']['stage1']
    static_cond_dim = config['static_cond_dim']
    seq_len = config['seq_len']
    gpu_device_ind = config['gpu_device_id']
    dataset_importer = DatasetImporterCustom(config=config, train_data_path=args.train_data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                           for kind in ['train', 'test']]
    train_stage1(config, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)
    model_config['data']['dataset'] = config['dataset']
    os.remove(new_model_config_save_path)
    with open(new_model_config_save_path, "w") as f:
        json.dump(model_config, f)

    # load training configs for Stage2
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    config = {}
    for d in (model_config['data'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)

    # Stage 2 training
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['dataset']['batch_sizes']['stage2']
    static_cond_dim = config['static_cond_dim']
    seq_len = config['seq_len']
    gpu_device_ind = config['gpu_device_id']
    dataset_importer = DatasetImporterCustom(config=config, train_data_path=args.train_data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind)
                                           for kind in ['train', 'test']]
    train_stage2(config, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
                 feature_extractor_type='rocket', use_custom_dataset=True)


def generate(args: argparse.Namespace):
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    config = {}
    for d in (model_config['general'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['evaluation']['batch_size']
    static_cond_dim = config['static_cond_dim']
    seq_len = config['seq_len']
    gpu_device_ind = config['gpu_device_id']
    dataset_importer = DatasetImporterCustom(config=config, train_data_path=None,
                                             test_data_path=args.static_cond_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['test']]
    static_conditions = torch.from_numpy(test_data_loader.dataset.SC)
    # generate synthetic data
    evaluate(config, dataset_name, static_cond_dim, dataset_importer, static_conditions, train_data_loader,
             gpu_device_ind,
             use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True)

    # clean memory
    torch.cuda.empty_cache()

    # concatenate conditions and TS data and convert them back to the source format
    cond_file_path = "./datasets/source_condition.csv"
    ts_file_path = os.path.join(f'synthetic_data', f'synthetic-{dataset_name}.csv')
    df_condition = pd.read_csv(cond_file_path)
    df_ts = pd.read_csv(ts_file_path)
    df_syn = pd.concat([df_condition, df_ts], axis=1)
    df_syn.to_csv(args.output_path + "synthetic_final.csv", index=False)




def BetterdataLogger(name='ml_logger', console_level='INFO', mode='MLOPS'):
    """
    Set up a logger with console, main file, and info-level file handlers.
    Optionally load configuration from JSON file based on mode.

    Parameters:
    - trace_log_filename: The filename for the debug and higher level log file. Defaults to 'logfile.txt'.
    - info_trace_log_filename: The filename for the info and higher level log file. Defaults to 'info_logfile.txt'.
    - console_level: The logging level for console output. Can be 'INFO', 'DEBUG', or 'TRACE'. Defaults to 'INFO'.
    - mode: The mode of operation, either 'local' or 'mlops'. Defaults to 'local'.

    Returns:
    - logger: Configured logger object.
    """

    if mode == 'MLOPS':
        json_path = '/tmp/ml_config.json'
        try:
            with open(json_path, 'r') as config_file:
                config = json.load(config_file)
                console_level = config.get('log_level', console_level)
        except FileNotFoundError:
            print(f"Configuration file not found at {json_path}. Using default settings.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON configuration file. Using default settings.")

    # Define TRACE level if not already defined
    if not hasattr(logging, 'TRACE'):
        logging.TRACE = 5
        logging.addLevelName(logging.TRACE, 'TRACE')

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.TRACE):
                self._log(logging.TRACE, message, args, **kwargs)

        logging.Logger.trace = trace

    # create special folder for logs
    path = './logs/'
    trace_log_filename = 'ml_trace_logfile.txt'
    info_log_filename = 'ml_info_logfile.txt'
    info_file_path = os.path.join(path, info_log_filename)
    trace_file_path = os.path.join(path, trace_log_filename)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(info_file_path):
        open(info_file_path, "w+").close()
    if not os.path.isfile(trace_file_path):
        open(trace_file_path, "w+").close()
        # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.TRACE)  # Set to TRACE to capture all logs

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(trace_file_path)  # Main file handler for debug and above
    info_f_handler = logging.FileHandler(info_file_path)  # Additional file handler for info and above

    # Set console level based on parameter
    if console_level.upper() == 'DEBUG':
        c_handler.setLevel(logging.DEBUG)
    elif console_level.upper() == 'TRACE':
        c_handler.setLevel(logging.TRACE)
    elif console_level.upper() == 'ERROR':
        c_handler.setLevel(logging.ERROR)
    else:
        c_handler.setLevel(logging.INFO)

    f_handler.setLevel(logging.TRACE)  # Set level to DEBUG for main file output
    info_f_handler.setLevel(logging.INFO)  # Set level to INFO for info file output

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    info_f_handler.setFormatter(info_f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.addHandler(info_f_handler)

    # Function to handle uncaught exceptions and log them
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Set the exception hook to our handler
    sys.excepthook = handle_exception

    # Function to handle fatal signals and log them
    def handle_signal(signum, frame):
        logger.error(f"Received signal {signum}")
        traceback.print_stack(frame)

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)  # Termination signal
    signal.signal(signal.SIGINT, handle_signal)  # Interrupt signal
    signal.signal(signal.SIGSEGV, handle_signal)  # Segmentation fault signal
    signal.signal(signal.SIGABRT, handle_signal)  # Abort signal
    signal.signal(signal.SIGFPE, handle_signal)  # Floating point exception signal
    signal.signal(signal.SIGILL, handle_signal)  # Illegal instruction signal

    return logger


def main():
    warnings.filterwarnings("ignore")
    args = prepare_args()
    all_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"]
    if args.log_level == all_levels[-1]:
        data_level = args.log_level
    else:
        data_level = all_levels[all_levels.index(args.log_level) + 1]
    BetterdataLogger("data-pipeline", data_level)
    BetterdataLogger("tabformer", args.log_level)
    # BetterdataLogger("arf", args.log_level)

    start_time = datetime.now()
    if args.op == "prepare":
        prepare(args)
    elif args.op == "validate":
        validate(args)
    elif args.op == "arf":
        warmup(args)
    elif args.op == "preprocess":
        preprocess(args)
    elif args.op == "train":
        train(args)
    elif args.op == "sample":
        generate(args)
    else:
        raise ValueError(f"Operation {args.op} is not recognized.")
    end_time = datetime.now()

    if args.time:
        diff = (end_time - start_time).total_seconds()
        if os.path.isdir(args.output_path):
            out = args.output_path
        else:
            out = os.path.dirname(args.output_path)
        with open(f"{out}-{args.op}-timing.json", "w") as f:
            json.dump({
                "Hr": diff // 60 // 60,
                "Mins": diff // 60 % 60,
                "Secs": diff % 60
            }, f)


if __name__ == "__main__":
    main()