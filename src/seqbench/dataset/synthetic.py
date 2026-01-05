# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Synthetic Dataset Module for SeqBench

This module provides simple embeddings such as one hot encoding.
"""

import numpy as np
import torch


class OneHot:
    """
    Synthetic one-hot encoded dataset.
    
    This class generates a simple dataset where each sample is a one-hot
    encoded vector. Useful for testing sequence models and data pipelines.
    
    Attributes:
        vocab_size: Number of unique tokens/classes in the vocabulary
        labels: Array of label indices (0 to vocab_size-1)
        stimulus: Dictionary mapping label indices to one-hot vectors
        class_dict: Dictionary mapping class indices to themselves
    """

    def __init__(self, vocab_size):
        """
        Initialize the one-hot dataset.
        
        Args:
            vocab_size: Number of unique tokens/classes in the vocabulary
        """

        self.vocab_size = vocab_size
        self.labels = np.arange(self.vocab_size)
        one_hot = np.eye(self.vocab_size, dtype=int)
        self.stimulus = {k: v for i, (k, v) in enumerate(zip(self.labels, one_hot))}
        x = np.arange(self.vocab_size)
        self.class_dict = {key: key for key in x}

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: The vocabulary size (number of unique tokens)
        """
        return self.vocab_size

    def __getitem__(self, index):
        """
        Get a one-hot encoded sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve (0 to vocab_size-1)
        
        Returns:
            tuple: (x, y) where:
                - x: One-hot encoded tensor of shape (1, vocab_size)
                - y: Label index (same as input index)
        """

        x = self.stimulus[index]
        x = x[None, :]

        return torch.from_numpy(x), index
