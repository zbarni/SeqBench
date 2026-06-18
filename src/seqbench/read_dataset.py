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

from seqbench import config as cfg_mod
from seqbench.utils import get_config_hash
from seqbench.seq_dataset import SeqDataset, make_pad_sequence
from seqbench.sources import build_symseq_source
from seqbench.dataset import create_base_dataset_from_config
from seqbench.transforms import compose_transforms_from_config

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('read_dataset')


def parse_cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
        help='The path to the .yaml which contains all user defined parameters.')
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_cli_arguments()
    run_cfg = cfg_mod.load(args['config'])

    assert run_cfg.seqbench is not None, "config must contain a 'seqbench' section"
    assert run_cfg.symseq is not None, "config must contain a 'symseq' section"

    seed = run_cfg.dataset.seed
    train_size = run_cfg.dataset.split_size('train')

    random.seed(seed)
    np.random.seed(seed)

    source = build_symseq_source(run_cfg)

    inp_map = run_cfg.seqbench.input_mapping

    kwargs = {}
    if inp_map.base == 'one_hot':
        kwargs['alphabet_size'] = len(source.alphabet)

    base_dataset = create_base_dataset_from_config(
        inp_map, 'train', dt=run_cfg.seqbench.dt, **kwargs
    )
    transforms = compose_transforms_from_config(inp_map)

    config_hash = get_config_hash(run_cfg, dataset_size=train_size)
    dataset_root = os.path.join(run_cfg.seqbench.storage.path, config_hash)

    seq_dataset = SeqDataset(
        config=run_cfg,
        generator=source,
        base_dataset=base_dataset,
        is_train=True,
        dataset_size=train_size,
        config_file_path=args['config'],
        pad_index=-1,
        dataset_root=dataset_root,
        transform=transforms,
    )

    seq_loader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=make_pad_sequence(seq_dataset, pad_index=-1),
        num_workers=1,
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
