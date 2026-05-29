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


def create_base_dataset_from_config(config_or_input_mapping, split, **kwargs):
    """
    Factory function to create a base dataset from configuration.

    Accepts either an :class:`~seqbench.config.InputMappingCfg` (new path) or
    a legacy ``Config`` object with an ``input_mapping`` key (old path).

    Args:
        config_or_input_mapping: InputMappingCfg or legacy Config
        split: Dataset split ('train' or 'test')
        **kwargs: Additional keyword arguments (e.g. ``alphabet_size`` for one_hot)
    """
    # New path: InputMappingCfg passed directly (has .base as a dataclass field).
    if hasattr(config_or_input_mapping, 'base'):
        return _create_from_input_mapping_cfg(config_or_input_mapping, split, **kwargs)
    # Old path: full Config with input_mapping key.
    return _create_from_legacy_config(config_or_input_mapping, split, **kwargs)


def _create_from_input_mapping_cfg(inp, split, **kwargs):
    """New path — inp is an InputMappingCfg; extra fields are in inp.base_params."""
    bp = inp.base_params  # plain dict

    if inp.base == "shd":
        return SpikingDataset(
            "shd",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if inp.base == "ssc":
        return SpikingDataset(
            "ssc",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if inp.base == "gsc":
        split = "training" if split == "train" else "testing"
        return SpeechCommands(
            data_folder=bp["base_dataset_path"],
            split=split,
            return_raw=bp.get("return_raw", False),
        )
    if inp.base == "one_hot":
        if "alphabet_size" not in kwargs:
            raise ValueError("alphabet_size must be provided for one_hot dataset!")
        return OneHot(kwargs["alphabet_size"] + 1)
    if inp.base.lower() in TONIC_DATASET_REGISTRY:
        canonical_name = TONIC_DATASET_REGISTRY[inp.base.lower()]
        return TonicDatasetWrapper(
            dataset_name=canonical_name,
            root=bp["base_dataset_path"],
            split=split,
            **{k: v for k, v in bp.items() if k != "base_dataset_path"},
        )
    raise ValueError(f"Unknown input encoding: {inp.base!r}")


def _create_from_legacy_config(config, split, **kwargs):
    """Old path — config is a full Config with an input_mapping key."""
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
            data_folder=os.path.join(inp_map_config["base_dataset_path"]),
            split=split,
            return_raw=inp_map_config["return_raw", False],
        )
    elif inp_map_config["base"] == "one_hot":
        if "alphabet_size" not in kwargs:
            raise ValueError("alphabet_size must be provided for one_hot dataset!")
        return OneHot(kwargs["alphabet_size"] + 1)
    elif inp_map_config["base"].lower() in TONIC_DATASET_REGISTRY:
        dataset_name_lower = inp_map_config["base"].lower()
        canonical_name = TONIC_DATASET_REGISTRY[dataset_name_lower]
        root = inp_map_config["base_dataset_path"]
        base_params = inp_map_config.get("base_params", {})
        return TonicDatasetWrapper(
            dataset_name=canonical_name,
            root=root,
            split=split,
            **base_params
        )
    else:
        raise ValueError(f"Unknown input encoding!")
