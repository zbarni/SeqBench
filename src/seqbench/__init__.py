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

from seqbench.dataset import create_base_dataset_from_config
from seqbench.seq_dataset import SeqDataset

__all__ = ['create_base_dataset_from_config', 'SeqDataset']
