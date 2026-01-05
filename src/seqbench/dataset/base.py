# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Base Dataset Module for SeqBench

This module provides the base dataset class that all SeqBench datasets inherit from.
It handles common functionality like class index extraction and caching.
"""

import os
import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for all datasets in SeqBench.
    
    This class provides common functionality for dataset management, including:
    - Automatic extraction and caching of class indexes
    - Support for various target formats (int, tensor, numpy array)
    - Path and split management
    
    Attributes:
        path_data: Path to the dataset directory
        inp_enc: Input encoding identifier (used for caching)
        split: Dataset split ('train' or 'test')
        class_dict: Dictionary mapping class indices to lists of sample indices
    
    Note:
        Class indexes are automatically cached in pickle files to speed up
        subsequent dataset loads. The cache file is named: {inp_enc}_{split}.pkl
    """

    def __init__(self, path, inp_enc, split):
        """
        Initialize the base dataset.
        
        Args:
            path: Path to the dataset directory
            inp_enc: Input encoding identifier (used for cache file naming)
            split: Dataset split ('train' or 'test')
        """
        self.path_data = path
        self.inp_enc = inp_enc
        self.split = split
        self.class_dict = self.extract_class_indexes()

    def extract_class_indexes(self):
        """
        Extract and cache class indexes from the dataset.
        
        This method creates a dictionary mapping each class index to a list of
        sample indices that belong to that class. The result is cached in a pickle
        file to avoid recomputation on subsequent loads.
        
        Returns:
            dict: Dictionary mapping class indices to lists of sample indices
            
        Raises:
            AttributeError: If the dataset doesn't return targets in a supported
                format (int, size-1 tensor, or size-1 numpy array)
        """
        sample0 = self[0]
        target0 = sample0[1]

        # Check if the value is already an integer
        if isinstance(target0, int):
            extract_int = lambda x: x
        elif isinstance(target0, np.integer):
            extract_int = lambda x: int(x)
        # Check if the value is a size-1 torch tensor with integer data
        elif isinstance(target0, torch.Tensor) and target0.numel() == 1 and target0.dtype in (torch.int32, torch.int64):
            extract_int = lambda x: x.item()
        # Check if the value is a size-1 numpy array with integer data
        elif isinstance(target0, np.ndarray) and target0.size == 1 and np.issubdtype(target0.dtype, np.integer):
            extract_int = lambda x: int(x.item())
        else:
            raise AttributeError("The dataset must return data that includes (sample, target, ...) and the target to be an integer, a size-1 tensor, or a size-1 array containing an integer. If this is not the case, please provide the class indexes dictionary manually.")

        # Initialize a default dictionary to hold lists of indexes for each class
        class_indexes = defaultdict(list)

        # Loop through all targets and append the index to the corresponding class key
        #for idx, sample in tqdm(enumerate(self), total=len(self), desc="Extracting class indexes"):
        #    class_indexes[extract_int(sample[1])].append(idx)
        

        try:
            with open(os.path.join(self.path_data, self.inp_enc + '_' + self.split) + '.pkl', "rb") as f:
                class_indexes = pickle.load(f)
                print('class_indexes loaded for:', self.inp_enc , "from:", self.path_data)
                #class_indexes = np.load(os.path.join(self.params['path_data'],
                #                                     self.params['inp_enc']) + '.npy', 
                #                        allow_pickle=True)
            return class_indexes
        except:    
            print('class_indexes doesnt exist for:', self.inp_enc + '_' + self.split)
            for idx, sample in tqdm(enumerate(self),
                                    total=len(self),
                                    desc="Extracting class indexes"):
                class_indexes[extract_int(sample[1])].append(idx)

            # Save it to a file
            os.makedirs(self.path_data, exist_ok=True)
            print('Saving class_indexes for:', self.inp_enc + '_' + self.split, "in:", self.path_data)
            with open(os.path.join(self.path_data, self.inp_enc + '_' + self.split) + '.pkl', "wb") as f:
                pickle.dump(class_indexes, f)
  
            return dict(class_indexes)
