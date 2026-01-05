# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Creation Script for SeqBench

This script creates sequence datasets from configuration files. It generates
both training and testing datasets based on the specified configuration.

Usage:
    python create_dataset.py --config path/to/config.yaml
"""

import os
import random
import logging
import argparse

import numpy as np

from seqbench.utils.config import Config
from seqbench.seq_dataset import DatasetGenerator
from seqbench import create_base_dataset_from_config
from seqbench.utils import prepare_config, get_config_hash
from seqbench.seq_utils.generator import SequenceGenerator

try:
    from symseq.seqwrapper import SeqWrapper
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('create_dataset')


def parse_cli_arguments():
    """
    Parse command-line arguments for the dataset creation script.
    
    Returns:
        dict: Dictionary containing parsed arguments with 'config' key
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the .yaml which contains all user defined parameters.')

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_cli_arguments()
    full_config = Config.parse_config_from_args(args)
    full_config.print_config()
    
    # Extract seqbench config
    config = full_config['seqbench']
    config['config_file_path'] = args['config']
    config = prepare_config(config)
    seed = config['seed']

    # Creating training dataset
    config['dataset_size'] = config['data_generation']['train_size']
    config_hash = get_config_hash(full_config)
    dataset_root = config['data_generation']['output_dir']
    dataset_root = f'{dataset_root}-{config_hash}'

    random.seed(seed)
    np.random.seed(seed)

    # Get sequencer from symseq
    if not HAS_SYMSEQ:
        raise ImportError("symseq is required for dataset generation")

    sw = SeqWrapper.from_dict(full_config)
    sequencer = sw.generator
    
    seq_generator = SequenceGenerator(config, sequencer)

    dataset_generator = DatasetGenerator(
        seq_generator,
        dataset_size=config['dataset_size'],
        output_dir=dataset_root,
        config_file_path=config['config_file_path'],
        generate_train=True,
        generate_test=False,
    )

    dataset_generator.generate()

    # Creating testing dataset ...
    config['dataset_size'] = config['data_generation']['test_size']
    config_hash = get_config_hash(full_config)
    dataset_root = config['data_generation']['output_dir']
    dataset_root = f'{dataset_root}-{config_hash}'

    random.seed(seed)
    np.random.seed(seed)

    seq_generator = SequenceGenerator(config, sequencer)

    dataset_generator = DatasetGenerator(
        seq_generator,
        dataset_size=config['dataset_size'],
        output_dir=dataset_root,
        config_file_path=config['config_file_path'],
        generate_train=False,
        generate_test=True,
    )
    
    dataset_generator.generate()