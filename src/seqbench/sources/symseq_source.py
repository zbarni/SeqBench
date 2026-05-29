# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
symseq trial source.

This is the only place in SeqBench that imports ``symseq``. Everywhere else,
the source is duck-typed against the structural ``TrialSource`` Protocol
(``alphabet``, ``draw_trial``, ``draw_batch``). Keeping the import here means
``import seqbench`` does not require symseq to be installed; only callers that
actually construct a symseq source pay the dependency.
"""

from __future__ import annotations

from typing import Any


def build_symseq_source(generator_cfg: Any) -> Any:
    """Build a live symseq generator from a config block.

    Parameters
    ----------
    generator_cfg
        A mapping shaped like ``{"type": <registry_name>, "params": {...}}``,
        typically pulled from a YAML/TOML config under ``symseq.generator``.
        Anything supporting ``__contains__`` / ``__getitem__`` / ``.get(...)``
        works (plain dict or seqbench's ``Config`` wrapper).

    Returns
    -------
    A ``symseq.core.sequencer.SymbolicSequencer`` subclass instance. The
    object structurally satisfies ``symseq.TrialSource`` (exposes
    ``alphabet``, ``draw_trial``, ``draw_batch``).
    """
    try:
        from symseq.generators.registry import build
    except ImportError as exc:
        raise ImportError(
            "Building a symseq trial source requires the `symseq` package. "
            "Install it, or supply a different TrialSource."
        ) from exc

    if "type" not in generator_cfg:
        raise ValueError(
            "symseq generator config requires a 'type' key (registry name)."
        )
    gen_type = generator_cfg["type"]
    params = generator_cfg.get("params", {}) or {}

    # Mirror symseq.config's special case: AG can be built from a named preset
    # (Elman, Reber, ...). Matches the schema accepted by symseq.load_trial_set.
    if gen_type == "ArtificialGrammar" and "preset" in generator_cfg:
        from symseq.generators.ag import ArtificialGrammar
        return ArtificialGrammar.from_preset(
            preset_name=generator_cfg["preset"],
            seed=params.get("seed", 42),
        )

    return build(gen_type, **params)
