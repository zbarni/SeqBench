# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utility functions for SeqBench.
"""

import numpy as np


def parse_gap_duration(identifier):
    if identifier == "uniform":
        return np.random.uniform
    elif identifier == "lognormal":
        return np.random.lognormal
    else:
        raise ValueError("Unknown gap_isi identifier!")


def get_config_hash(run_cfg, dataset_size=None):
    """Compute a cache key hash from a :class:`~seqbench.config.RunConfig`.

    Pass ``dataset_size`` explicitly to include the split size in the hash
    (used to differentiate train vs. test dataset directories).
    """
    return _get_config_hash_from_run_cfg(run_cfg, dataset_size)


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
