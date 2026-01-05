# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Reading and Debugging Script for SeqBench

This script loads and displays dataset samples for debugging and inspection purposes.
It creates a dataset from a configuration file and prints sample information.

Usage:
    python read_dataset.py --config path/to/config.yaml
"""

import os
import random
import logging
import argparse

import torch
import numpy as np

from seqbench.utils.config import Config
from seqbench.utils import prepare_config
from seqbench.seq_dataset import PadSequence

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('read_dataset')

def parse_cli_arguments():
    """
    Parse command-line arguments for the dataset reading script.
    
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
    config['do_classify'] = True

    random.seed(seed)
    np.random.seed(seed)

    config['dataset_size'] = config['data_generation']['train_size']
    
    # For create_seq_dataset_from_config, we need to pass the full config
    # but it will extract seqbench internally. However, for one_hot we need alphabet_size
    # So we'll create the dataset manually similar to create_dataset.py
    try:
        from symseq.seqwrapper import SeqWrapper
        sw = SeqWrapper.from_dict(full_config)
        generator = sw.generator
    except ImportError:
        raise ImportError("symseq is required")
    
    # Create base dataset with alphabet_size for one_hot
    kwargs = {}
    if config['input_mapping']['base'] == 'one_hot':
        if 'alphabet_size' in full_config.get('symseq', {}).get('generator', {}).get('constraints', {}):
            kwargs['alphabet_size'] = full_config['symseq']['generator']['constraints']['alphabet_size']
    
    from seqbench.dataset import create_base_dataset_from_config
    from seqbench.transforms import compose_transforms_from_config
    from seqbench.utils import get_config_hash
    
    base_dataset = create_base_dataset_from_config(config, 'train', **kwargs)
    
    # Compose transforms
    transforms = compose_transforms_from_config(config)
    
    # Get dataset root
    config_hash = get_config_hash(full_config)
    dataset_root = config['data_generation']['output_dir']
    dataset_root = f"{dataset_root}/{config_hash}"
    
    from seqbench.seq_dataset import SeqDataset
    seq_dataset = SeqDataset(
        config=config,
        generator=generator,
        base_dataset=base_dataset,
        is_train=True,
        pad_index=-1,
        dataset_root=dataset_root,
        transform=transforms,
    )

    seq_loader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=PadSequence(
            do_classify=config['do_classify'],
            pad_index=-1
        ),
        num_workers=1
    )  

    for entry in seq_loader:
        print('~~~')
        print(entry['data'].shape)
        print(entry['labels'].shape)
        print('min label', torch.min(entry['labels']))
        print('max label', torch.max(entry['labels']))
        print(entry['mask'].shape)
        print(entry['lens'].shape)
        print(entry['gap_mask'].shape)
