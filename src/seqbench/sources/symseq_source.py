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


def build_symseq_source(config: Any) -> Any:
    """Build a live symseq generator from a config block.

    Parameters
    ----------
    config
        Either a :class:`~seqbench.config.RunConfig`, a
        :class:`~seqbench.config.GeneratorCfg` dataclass instance, or a plain
        dict shaped like ``{"type": <registry_name>, "params": {...}}``.

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

    generator_cfg, dataset_cfg, seed = _normalise_config(config)

    # Normalise: accept a GeneratorCfg dataclass or a plain dict.
    if isinstance(generator_cfg, dict):
        gen_type = generator_cfg.get("type")
        params = dict(generator_cfg.get("params") or {})
        # Legacy symseq standalone schema: top-level "preset" key in the dict.
        ag_preset = generator_cfg.get("preset")
        ag_mode = _enum_value(generator_cfg.get("mode"))
    else:
        # GeneratorCfg dataclass: fields accessed as attributes.
        gen_type = generator_cfg.type
        params = dict(generator_cfg.params)
        ag_preset = None
        ag_mode = _enum_value(generator_cfg.mode)  # AGMode enum or None

    if not gen_type:
        raise ValueError(
            "symseq generator config requires a 'type' key (registry name)."
        )

    # Special case: ArtificialGrammar preset constructor.
    if gen_type == "ArtificialGrammar":
        from symseq.generators.ag import ArtificialGrammar

        if ag_mode == "preset":
            # Dataclass path: preset name lives in params["preset"].
            preset_name = params.pop("preset", None)
            if not preset_name:
                raise ValueError(
                    "ArtificialGrammar with mode='preset' requires a 'preset' key "
                    "inside generator.params (e.g. params: {preset: Elman})."
                )
            return ArtificialGrammar.from_preset(
                preset_name=preset_name,
                seed=params.get("seed", seed if seed is not None else 42),
            )
        if ag_preset:
            # Legacy dict path: top-level "preset" key.
            return ArtificialGrammar.from_preset(
                preset_name=ag_preset,
                seed=params.get("seed", seed if seed is not None else 42),
            )
        if ag_mode in (None, "random"):
            _inherit_ag_random_defaults(params, dataset_cfg, seed)
            return ArtificialGrammar.from_constraints(**params)

    if seed is not None and "seed" not in params and "rng" not in params:
        params["seed"] = seed

    return build(gen_type, **params)


def _normalise_config(config: Any) -> tuple[Any, Any | None, int | None]:
    """Return ``(generator_cfg, dataset_cfg, effective_seed)``.

    ``RunConfig`` is preferred because it gives this boundary enough context to
    apply documented dataset -> symseq inheritance. Bare generator configs still
    work for older call sites and tests.
    """
    if hasattr(config, "symseq") and hasattr(config, "dataset"):
        if config.symseq is None:
            raise ValueError("RunConfig must contain a 'symseq' section")
        seed = config.symseq.seed if config.symseq.seed is not None else config.dataset.seed
        return config.symseq.generator, config.dataset, seed
    return config, None, None


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _inherit_ag_random_defaults(
    params: dict[str, Any],
    dataset_cfg: Any | None,
    seed: int | None,
) -> None:
    if dataset_cfg is not None:
        params.setdefault("alphabet_size", dataset_cfg.alphabet.size)
        if dataset_cfg.alphabet.eos is not None:
            params.setdefault("eos", dataset_cfg.alphabet.eos)
    if seed is not None:
        params.setdefault("seed", seed)

