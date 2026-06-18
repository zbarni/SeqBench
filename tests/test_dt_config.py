# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Tests for the global grid resolution ``seqbench.dt``: default, validation,
and the back-compat shim that lifts a legacy ``gap_profile.dt`` to ``seqbench.dt``."""

import pytest

from seqbench import config as cfg_mod


def _raw(**seqbench_overrides):
    """Minimal valid config; seqbench section is shallow-merged with overrides."""
    seqbench = {
        "mode": "online",
        "prob_generator_type": "restricted",
        "storage": {"path": "/tmp/sb_dt_test"},
        "composition": {"combine_sequences": False, "sample_length": 20},
        "input_mapping": {"base": "one_hot", "base_params": {}, "transforms": []},
        "task": {"source": "seqbench", "type": "StateClassification"},
    }
    seqbench.update(seqbench_overrides)
    return {
        "dataset": {
            "seed": 1,
            "alphabet": {"size": 4, "eos": "#"},
            "trial_length": {"min": 1, "max": 20},
            "splits": {"train": 8, "test": 4},
        },
        "seqbench": seqbench,
    }


def test_dt_defaults_to_point_one():
    cfg = cfg_mod.load(_raw())
    assert cfg.seqbench.dt == 0.1


def test_dt_explicit_top_level():
    cfg = cfg_mod.load(_raw(dt=0.05))
    assert cfg.seqbench.dt == 0.05


@pytest.mark.parametrize("bad", [0, -0.1, "x"])
def test_dt_must_be_positive(bad):
    with pytest.raises(ValueError, match="dt must be a positive"):
        cfg_mod.load(_raw(dt=bad))


def test_legacy_gap_profile_dt_is_lifted_with_warning():
    raw = _raw(
        composition={
            "combine_sequences": False,
            "sample_length": 20,
            "gap_profile": {
                "start": 0,
                "duration": {"dist": "uniform", "params": {"low": 5, "high": 5}},
                "add_nongramm_gap": False,
                "dt": 0.2,
            },
        }
    )
    with pytest.warns(DeprecationWarning, match="gap_profile.dt is deprecated"):
        cfg = cfg_mod.load(raw)
    assert cfg.seqbench.dt == 0.2
    # The lifted value must not survive on the gap profile.
    assert not hasattr(cfg.seqbench.composition.gap_profile, "dt")


def test_top_level_dt_wins_over_legacy_gap_profile_dt():
    raw = _raw(
        dt=0.1,
        composition={
            "combine_sequences": False,
            "sample_length": 20,
            "gap_profile": {
                "start": 0,
                "duration": {"dist": "uniform", "params": {"low": 5, "high": 5}},
                "add_nongramm_gap": False,
                "dt": 0.2,
            },
        },
    )
    with pytest.warns(DeprecationWarning, match="ignoring gap_profile.dt"):
        cfg = cfg_mod.load(raw)
    assert cfg.seqbench.dt == 0.1
