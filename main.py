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
from tabtransformer import TableTransformer
from ARFSynthesizer import ARFSynthesizer
from arflib import utils
import pandas as pd
import numpy as np

# import functions from original developed TS repo
from stage1 import *
from stage2 import *
from generate import *



def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", "-l", type=str, default="INFO",
                        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"])
    parser.add_argument("--time", "-t", action="store_true")
    subparsers = parser.add_subparsers(dest="op")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data-path", "-d", required=True)
    prepare_parser.add_argument("--data-policy", "-p", type=str,
                                default="./configs/data/gan_core.json")
    prepare_parser.add_argument("--output-path", "-o", default="./out/ts_data_config_learn.json")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--data-path", "-d", required=True)
    validate_parser.add_argument("--sample-size", "-s", default=50000)
    validate_parser.add_argument("--data-config", "-p", type=str,
                                 default="./out/ts_data_config_learn.json")
    validate_parser.add_argument("--output-path", "-o", default="./out/ts_data_config_learn.json")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-path", "-d", required=True)
    train_parser.add_argument("--test-data-path", "-t", required=True)
    train_parser.add_argument("--data-config", "-c", type=str,
                              default="./out/ts_data_config_learn.json")
    train_parser.add_argument("--model-config", "-m", type=str,
                              default="./configs/config.json")
    train_parser.add_argument("--model-saving-path", "-s", type=str,
                              default="./out/ts_model_config.json")
    train_parser.add_argument("--output-path", "-o", default="./out")

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--model-loading-path", "-l", type=str,
                               default="./out/ts_model_config.json")
    sample_parser.add_argument("--model-config", "-m", type=str,
                               default="./configs/config.json")
    sample_parser.add_argument("--output-path", "-o", default="./synthetic_data/generated.csv")
    return parser.parse_args()


def prepare(args: argparse.Namespace):
    with open(args.data_policy, "r") as f:
        data_config = json.load(f)

    # Load original dataset ?
    data = args.data_path
    df_data = pd.read_csv(data)


    # Define the proper data pipeline for your model
    TableTransformer.validate_kwargs(df_data, data_config, learning=True)
    learned_args = TableTransformer.learn_args(df_data, **data_config)
    TableTransformer.validate_kwargs(df_data, learned_args, learning=False)

    directory = os.path.dirname(args.output_path)
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    with open(args.output_path, "w") as f:
        json.dump(learned_args, f, indent=2)


def validate(args: argparse.Namespace):
    data = args.data_path
    df_data = pd.read_csv(data)
    with open(args.data_config, "r") as f:
        data_config = json.load(f)

    TableTransformer.validate_kwargs(df_data, data_config, learning=False)


def train(args: argparse.Namespace):
    # load training configs
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    config = {}
    for d in (model_config['general'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['dataset']['batch_sizes']['stage1']
    static_cond_dim = config['dataset']['static_cond_dim']
    seq_len = config['dataset']['seq_len']
    gpu_device_ind = config['gpu_device_id']
    dataset_importer = DatasetImporterCustom(train_data_path=args.data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for
                                           kind in ['train', 'test']]
    # Stage 1 training
    train_stage1(config, dataset_name, train_data_loader, test_data_loader, gpu_device_ind)

    # Stage 2 training
    train_stage2(config, dataset_name, static_cond_dim, train_data_loader, test_data_loader, gpu_device_ind,
                 feature_extractor_type='rocket', use_custom_dataset=True)


def generate(args: argparse.Namespace):
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    config = {}
    for d in (model_config['general'], model_config['train'], model_config['model'], model_config['generate']): config.update(d)
    dataset_name = config['dataset']['dataset_name']
    batch_size = config['evaluation']['batch_size']
    static_cond_dim = config['dataset']['static_cond_dim']
    seq_len = config['dataset']['seq_len']
    gpu_device_ind = config['gpu_device_id']
    dataset_importer = DatasetImporterCustom(train_data_path=args.data_path,
                                             test_data_path=args.test_data_path, static_cond_dim=static_cond_dim,
                                             seq_len=seq_len, **config['dataset'])
    train_data_loader, test_data_loader = [build_custom_data_pipeline(batch_size, dataset_importer, config, kind) for
                                           kind in ['train', 'test']]
    static_conditions = torch.from_numpy(test_data_loader.dataset.SC)
    # generate synthetic data
    evaluate(config, dataset_name, static_cond_dim, dataset_importer, static_conditions, train_data_loader,
             gpu_device_ind,
             use_fidelity_enhancer=False, feature_extractor_type='rocket', use_custom_dataset=True)

    # clean memory
    torch.cuda.empty_cache()


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
    BetterdataLogger("arf", args.log_level)

    start_time = datetime.now()
    if args.op == "prepare":
        prepare(args)
    elif args.op == "validate":
        validate(args)
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