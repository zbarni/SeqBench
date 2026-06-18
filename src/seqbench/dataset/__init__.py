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

import warnings

from seqbench.dataset.shd_ssc import SpikingDataset
from seqbench.dataset.speech_commands import SpeechCommands
from seqbench.dataset.synthetic import OneHot
from seqbench.dataset.tonic_wrapper import TonicDatasetWrapper, TONIC_DATASET_REGISTRY


def _warn_native_dt_mismatch(base, nb_steps, max_time, dt):
    """Warn if the global grid ``dt`` differs from a spiking dataset's native dt.

    SHD/SSC bin spikes over ``max_time`` into ``nb_steps`` bins, fixing their
    native resolution at ``max_time / nb_steps``. Under the single real-time
    axis the global ``dt`` should match this; resampling is a future step, so
    for now we only surface the mismatch.
    """
    if dt is None or not nb_steps:
        return
    native_dt = max_time / nb_steps
    if abs(native_dt - dt) > 1e-9:
        warnings.warn(
            f"{base!r} has a native dt of {native_dt:g}s "
            f"(max_time={max_time}/nb_steps={nb_steps}) but seqbench.dt={dt:g}s. "
            "These stimuli are not resampled to the global grid; the time axis "
            "will be inconsistent until dt matches the native resolution.",
            stacklevel=2,
        )


def create_base_dataset_from_config(input_mapping, split, dt=None, **kwargs):
    """
    Factory function to create a base dataset from an
    :class:`~seqbench.config.InputMappingCfg`.

    Args:
        input_mapping: :class:`~seqbench.config.InputMappingCfg` instance.
        split: Dataset split ('train' or 'test').
        dt: Global grid resolution in seconds (``seqbench.dt``). Used to convert
            ``base_params.duration`` (seconds) into a stimulus step count for
            one_hot, and to validate spiking datasets' native resolution.
        **kwargs: Additional keyword arguments (e.g. ``alphabet_size`` for one_hot).
    """
    bp = input_mapping.base_params  # plain dict of base-specific parameters

    if input_mapping.base == "shd":
        _warn_native_dt_mismatch("shd", bp["nb_steps"], bp["max_time"], dt)
        return SpikingDataset(
            "shd",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if input_mapping.base == "ssc":
        _warn_native_dt_mismatch("ssc", bp["nb_steps"], bp["max_time"], dt)
        return SpikingDataset(
            "ssc",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if input_mapping.base == "gsc":
        split = "training" if split == "train" else "testing"
        return SpeechCommands(
            data_folder=bp["base_dataset_path"],
            split=split,
            return_raw=bp.get("return_raw", False),
        )
    if input_mapping.base == "one_hot":
        if "alphabet_size" not in kwargs:
            raise ValueError("alphabet_size must be provided for one_hot dataset!")
        # Stimulus footprint = round(duration / dt) steps. Without a global dt
        # (legacy callers) fall back to a single timestep. ``duration`` defaults
        # to ``dt`` so an unspecified stimulus is exactly one step.
        if dt is not None:
            duration = bp.get("duration", dt)
            n_steps = max(1, round(duration / dt))
        else:
            n_steps = 1
        return OneHot(kwargs["alphabet_size"] + 1, n_steps=n_steps)
    if input_mapping.base.lower() in TONIC_DATASET_REGISTRY:
        canonical_name = TONIC_DATASET_REGISTRY[input_mapping.base.lower()]
        return TonicDatasetWrapper(
            dataset_name=canonical_name,
            root=bp["base_dataset_path"],
            split=split,
            **{k: v for k, v in bp.items() if k != "base_dataset_path"},
        )
    raise ValueError(f"Unknown input encoding: {input_mapping.base!r}")
