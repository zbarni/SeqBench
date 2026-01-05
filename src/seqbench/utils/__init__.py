# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utility functions for configuration parsing and dataset preparation.
"""

import numpy as np

from seqbench.utils.config import Config


def parse_gap_duration(identifier):
    if identifier == "uniform":
        return np.random.uniform
    elif identifier == "lognormal":
        return np.random.lognormal
    else:
        raise ValueError("Unknown gap_isi identifier!")


def prepare_config(config):
    # config['gramm']['transition_density'] /= config['gramm']['ambiguity_depth']
    config = parse_gap_profile(config)
    config = parse_timestep(config)
    config = parse_transforms(config)
    return config


def parse_transforms(config):
    # rest is done in the transform module
    if "transforms" not in config["input_mapping"]:
        config["transforms"] = None

    return config


def parse_gap_profile(config):
    if "gap_profile" in config:
        gap_profile = config["gap_profile"]
        if "dist" in dict(gap_profile["duration"]):
            gap_profile["duration"]["dist"] = parse_gap_duration(gap_profile["duration"]["dist"])
    else:
        # TODO maybe fill with defaults here?
        config["gap_profile"] = None

    return config


# TODO @Younes - this needs to be updated!
def parse_timestep(config):
    # Infer timestep from max_time and nb_steps if not provided
    # if "timestep" not in dict(config):
    #     if "base" in config and config["base"] in ("shd", "ssc"):
    #         if "max_time" in config and "nb_steps" in config:
    #             config.config["timestep"] = config["max_time"] * 1000 / config["nb_steps"]
    #     else:
    #         # timestep might be explicitly set or not needed for this base type
    #         pass

    return config


def get_config_hash(config):
    import hashlib
    from pprint import pformat

    key = {}

    # Check if this is a full config (with symseq/seqbench) or a seqbench-only config
    if "symseq" in config:
        # Full config structure
        key["generator"] = config["symseq"]["generator"].asdict()
        seqbench_config = config["seqbench"]
    else:
        # seqbench-only config (e.g., after extracting config['seqbench'])
        key["generator"] = {}
        seqbench_config = config

    # Access data_generation parameters with nested structure
    data_gen = seqbench_config["data_generation"]
    key["seq_len_max"] = data_gen["seq_len_max"]
    key["seq_len_min"] = data_gen["seq_len_min"]
    key["combined_seq_length"] = data_gen["combined_seq_length"]

    # dataset_size is set dynamically (either train_size or test_size)
    # so check if it exists, otherwise don't include in hash
    if "dataset_size" in seqbench_config:
        key["dataset_size"] = seqbench_config["dataset_size"]

    key["seed"] = seqbench_config["seed"]

    config_hash = hashlib.md5(pformat(key).encode("utf-8")).hexdigest()

    assert config_hash is not None

    return config_hash
