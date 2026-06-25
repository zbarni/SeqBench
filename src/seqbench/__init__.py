# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
SeqBench: A Python library for transforming, embedding symbolic sequences.

This package provides tools for:
- Creating and managing sequence datasets
- Supporting diverse embeddings and mappings to datasets
- Classification tasks with various input encodings (SHD, GSC, etc.)
- Transform pipelines for data preprocessing
"""

__all__ = ['create_base_dataset_from_config', 'initial_time_grid_from_config', 'SeqDataset']


def __getattr__(name):
    if name in {"create_base_dataset_from_config", "initial_time_grid_from_config"}:
        from seqbench.dataset import create_base_dataset_from_config, initial_time_grid_from_config

        values = {
            "create_base_dataset_from_config": create_base_dataset_from_config,
            "initial_time_grid_from_config": initial_time_grid_from_config,
        }
        return values[name]
    if name == "SeqDataset":
        from seqbench.seq_dataset import SeqDataset

        return SeqDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
