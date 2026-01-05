# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Module for SeqBench

This module provides dataset classes and factory functions for creating base datasets
from configuration files. Supported datasets include:
- SHD (Spiking Heidelberg Digits)
- SSC (Spiking Speech Commands)
- GSC (Google Speech Commands)
- One-hot synthetic datasets
- Tonic datasets (DVSGesture, NMNIST, etc.)
"""

import os
from seqbench.dataset.shd_ssc import SpikingDataset
from seqbench.dataset.speech_commands import SpeechCommands
from seqbench.dataset.synthetic import OneHot
from seqbench.dataset.tonic_wrapper import TonicDatasetWrapper, TONIC_DATASET_REGISTRY


def create_base_dataset_from_config(config, split, **kwargs):
    """
    Factory function to create a base dataset from configuration.
    
    This function reads the input mapping configuration and instantiates the
    appropriate dataset class based on the 'base' field in the config.
    
    Args:
        config: Configuration object containing input_mapping settings
        split: Dataset split ('train' or 'test')
        **kwargs: Additional keyword arguments passed to dataset constructors
            (e.g., 'alphabet_size' for one_hot datasets)
    
    Returns:
        BaseDataset: An instance of the appropriate dataset class
        
    Raises:
        ValueError: If the input encoding type is unknown or required parameters
            are missing (e.g., alphabet_size for one_hot datasets)
    
    Supported base types:
        - 'shd': Spiking Heidelberg Digits dataset
        - 'ssc': Spiking Speech Commands dataset
        - 'gsc': Google Speech Commands dataset
        - 'one_hot': Synthetic one-hot encoded dataset
        - Any key in TONIC_DATASET_REGISTRY: Tonic library datasets
    """
    inp_map_config = config["input_mapping"]
    if inp_map_config["base"] == "shd":
        return SpikingDataset(
            "shd",
            inp_map_config["base_dataset_path"],
            split,
            nb_steps=inp_map_config["nb_steps"],
            max_time=inp_map_config["max_time"],
            num_bins=inp_map_config["num_bins", 1],
        )
    if inp_map_config["base"] == "ssc":
        return SpikingDataset(
            "ssc",
            inp_map_config["base_dataset_path"],
            split,
            nb_steps=inp_map_config["nb_steps"],
            max_time=inp_map_config["max_time"],
            num_bins=inp_map_config["num_bins", 1],
        )
    elif inp_map_config["base"] == "gsc":
        if split == "train":
            split = "training"
        else:
            split = "testing"

        method = "mfcc"
        if "base_params" in inp_map_config and "method" in inp_map_config["base_params"]:
            method = inp_map_config["base_params"]["method"]

        return SpeechCommands(
            # os.path.join(inp_map_config['base_dataset_path'], 'GSC'),
            data_folder=os.path.join(inp_map_config["base_dataset_path"]),
            split=split,
            # method=method,  @Younes - double check what happened to this param
            return_raw=inp_map_config["return_raw", False],
        )

    elif inp_map_config["base"] == "one_hot":
        if "alphabet_size" not in kwargs:
            raise ValueError("alphabet_size must be provided for one_hot dataset!")
        base_dataset = OneHot(kwargs["alphabet_size"] + 1)  # +1 for the end sequence symbol

        return base_dataset
    
    elif inp_map_config["base"].lower() in TONIC_DATASET_REGISTRY:
        # Handle tonic datasets
        dataset_name_lower = inp_map_config["base"].lower()
        canonical_name = TONIC_DATASET_REGISTRY[dataset_name_lower]
        root = inp_map_config["base_dataset_path"]
        
        # Get additional parameters from base_params if provided
        base_params = inp_map_config.get("base_params", {})
        
        return TonicDatasetWrapper(
            dataset_name=canonical_name,
            root=root,
            split=split,
            **base_params
        )

    else:
        raise ValueError(f"Unknown input encoding!")
