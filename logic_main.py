import os
import yaml
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def update_args_from_config(args, config):
    for key, value in config.items():
        setattr(args, key, value)

def setup_logging(args):
    log_file_path = f"{args.log_dir}/{args.exp}/config.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger(__name__)
    # Set the log level globally for all handlers
    logger.setLevel(logging.INFO)
    
    # If logger already has handlers, remove all (to avoid duplicate logging)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler which logs messages
    file_handler = logging.FileHandler(log_file_path, mode='w')  # 'w' to overwrite the log file on each run
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)

    return logger

def main(args):
    # Set up logging
    os.makedirs(f'{args.log_dir}/{args.exp}', exist_ok=True)
    logger = setup_logging(args)
    writer = SummaryWriter(f'{args.log_dir}/{args.exp}')
    # start running
    if "Neural" in args.model_type:
        from tasks.logic.neural_runner import runner
        runner(args, logger, writer)
    elif "Symbolic" in args.model_type:
        from tasks.logic.symbolic_runner import runner
        runner(args, logger, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/tasks/logic/easy/small_symbolicdt.yaml', help='Directory to configure file')
    parser.add_argument('--resume', default=False, help='Resume training from a checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for the dataloader')
    parser.add_argument('--data_path', type=str, default='log/easy_1k_train.pkl', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='log/easy_100_test.pkl', help='Path to the test data')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Update args with config
    update_args_from_config(args, config)
    main(args)
