# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utility functions for configuration parsing and dataset preparation.
"""

import numpy as np

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


def get_config_hash(config, dataset_size=None):
    """Compute a cache key hash from a config.

    Accepts either a :class:`~seqbench.config.RunConfig` (new path) or the
    legacy ``Config`` wrapper (old path, used by seq_dataset.py).  When using
    ``RunConfig``, pass ``dataset_size`` explicitly so it can be included in
    the hash without mutating the config.
    """
    from seqbench.config import RunConfig
    if isinstance(config, RunConfig):
        return _get_config_hash_from_run_cfg(config, dataset_size)
    return _get_config_hash_from_legacy(config)


def _get_config_hash_from_run_cfg(run_cfg, dataset_size=None):
    import hashlib
    from dataclasses import asdict
    from pprint import pformat

    key = {}
    if run_cfg.symseq is not None:
        gen = asdict(run_cfg.symseq.generator)
        # Normalise enum to its string value so the hash is stable across
        # Python sessions and independent of enum repr changes.
        if gen.get("mode") is not None:
            gen["mode"] = str(gen["mode"].value) if hasattr(gen["mode"], "value") else str(gen["mode"])
        key["generator"] = gen
    else:
        key["generator"] = {}

    key["seq_len_max"] = run_cfg.dataset.trial_length.max
    key["seq_len_min"] = run_cfg.dataset.trial_length.min
    key["combined_seq_length"] = run_cfg.seqbench.composition.sample_length
    if dataset_size is not None:
        key["dataset_size"] = dataset_size
    key["seed"] = run_cfg.dataset.seed

    return hashlib.md5(pformat(key).encode("utf-8")).hexdigest()


def _get_config_hash_from_legacy(config):
    import hashlib
    from pprint import pformat

    key = {}

    if "symseq" in config:
        key["generator"] = config["symseq"]["generator"].asdict()
        seqbench_config = config["seqbench"]
    else:
        key["generator"] = {}
        seqbench_config = config

    data_gen = seqbench_config["data_generation"]
    key["seq_len_max"] = data_gen["seq_len_max"]
    key["seq_len_min"] = data_gen["seq_len_min"]
    key["combined_seq_length"] = data_gen["combined_seq_length"]

    if "dataset_size" in seqbench_config:
        key["dataset_size"] = seqbench_config["dataset_size"]

    key["seed"] = seqbench_config["seed"]

    return hashlib.md5(pformat(key).encode("utf-8")).hexdigest()
