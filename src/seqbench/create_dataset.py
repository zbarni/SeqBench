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

from seqbench import config as cfg_mod
from seqbench.seq_dataset import DatasetGenerator
from seqbench.utils import get_config_hash
from seqbench.seq_utils.generator import SequenceGenerator
from seqbench.sources import build_symseq_source

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('create_dataset')


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
    train_size = int(run_cfg.dataset.splits['train'])
    test_size = int(run_cfg.dataset.splits['test'])
    output_dir = run_cfg.seqbench.storage.path

    source = build_symseq_source(run_cfg.symseq.generator)

    seq_generator = SequenceGenerator(
        source,
        seq_len_min=run_cfg.dataset.trial_length.min,
        seq_len_max=run_cfg.dataset.trial_length.max,
        combine_sequences=run_cfg.seqbench.composition.combine_sequences,
        combined_seq_len=run_cfg.seqbench.composition.sample_length,
        seed=seed,
    )

    # Creating training dataset
    random.seed(seed)
    np.random.seed(seed)

    config_hash = get_config_hash(run_cfg, dataset_size=train_size)
    dataset_root = f'{output_dir}-{config_hash}'

    dataset_generator = DatasetGenerator(
        seq_generator,
        dataset_size=train_size,
        output_dir=dataset_root,
        config_file_path=args['config'],
        generate_train=True,
        generate_test=False,
    )
    dataset_generator.generate()

    # Creating testing dataset
    random.seed(seed)
    np.random.seed(seed)

    config_hash = get_config_hash(run_cfg, dataset_size=test_size)
    dataset_root = f'{output_dir}-{config_hash}'

    seq_generator = SequenceGenerator(
        source,
        seq_len_min=run_cfg.dataset.trial_length.min,
        seq_len_max=run_cfg.dataset.trial_length.max,
        combine_sequences=run_cfg.seqbench.composition.combine_sequences,
        combined_seq_len=run_cfg.seqbench.composition.sample_length,
        seed=seed,
    )

    dataset_generator = DatasetGenerator(
        seq_generator,
        dataset_size=test_size,
        output_dir=dataset_root,
        config_file_path=args['config'],
        generate_train=False,
        generate_test=True,
    )
    dataset_generator.generate()
