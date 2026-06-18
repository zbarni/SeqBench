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
from seqbench.dataset_generator import RestrictedTargetProbGenerator
from seqbench.sources import build_symseq_source
from seqbench.tasks.target_builder import make_task_target_builder

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
    train_size = run_cfg.dataset.split_size('train')
    test_size = run_cfg.dataset.split_size('test')
    output_dir = run_cfg.seqbench.storage.path

    source = build_symseq_source(run_cfg)

    def build_seq_generator():
        """Construct a SequenceGenerator with the configured task builder wired in.

        The builder resolves and serializes the training target per sample; a
        transition-loaded prob generator is supplied so grammar-dependent tasks
        (e.g. StateClassification) can resolve at draw time.
        """
        gen = SequenceGenerator(
            source,
            seq_len_min=run_cfg.dataset.trial_length.min,
            seq_len_max=run_cfg.dataset.trial_length.max,
            combine_sequences=run_cfg.seqbench.composition.combine_sequences,
            combined_seq_len=run_cfg.seqbench.composition.sample_length,
            seed=seed,
            trial_params=run_cfg.symseq.generator.trial_params,
        )
        prob_gen = None
        if hasattr(source, "transitions"):
            prob_gen = RestrictedTargetProbGenerator()
            prob_gen.read_transitions_from_generator(gen)
        gen.task_builder = make_task_target_builder(run_cfg, source, prob_generator=prob_gen)
        return gen

    # Creating training dataset
    random.seed(seed)
    np.random.seed(seed)

    config_hash = get_config_hash(run_cfg, dataset_size=train_size)
    dataset_root = os.path.join(output_dir, config_hash)

    dataset_generator = DatasetGenerator(
        build_seq_generator(),
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
    dataset_root = os.path.join(output_dir, config_hash)

    dataset_generator = DatasetGenerator(
        build_seq_generator(),
        dataset_size=test_size,
        output_dir=dataset_root,
        config_file_path=args['config'],
        generate_train=False,
        generate_test=True,
    )
    dataset_generator.generate()
