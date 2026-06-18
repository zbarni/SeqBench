# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Typed schema for the unified symseq + SeqBench YAML config.

Parses a YAML file (or pre-loaded dict) into a :class:`RunConfig` of nested
dataclasses, validating structural constraints by hand (no pydantic). The
schema is documented in detail in
``SeqBench/examples/configs/_schema_reference.yaml``.

This module is intentionally side-effect-free: it does NOT perform
``dataset → symseq`` inheritance, does NOT instantiate generators, and does
NOT touch disk. Those steps belong to the loader that consumes a parsed
:class:`RunConfig` (next step).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ----------------------------- enums ------------------------------------------


class OperatingMode(str, Enum):
    SYMSEQ_ONLY = "symseq_only"
    SYMSEQ_PLUS_SEQBENCH = "symseq_plus_seqbench"
    SEQBENCH_STANDALONE = "seqbench_standalone"


class AGMode(str, Enum):
    RANDOM = "random"
    PRESET = "preset"
    CUSTOM = "custom"


class TaskSource(str, Enum):
    SYMSEQ = "symseq"
    SEQBENCH = "seqbench"


# ----------------------------- dataset ----------------------------------------


@dataclass
class AlphabetCfg:
    size: int
    eos: str | None = "#"

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError(f"alphabet.size must be a positive int, got {self.size!r}")
        if self.eos is not None and not isinstance(self.eos, str):
            raise ValueError(f"alphabet.eos must be a string or null, got {self.eos!r}")


@dataclass
class TrialLengthCfg:
    min: int
    max: int
    distribution: str = "uniform"

    def __post_init__(self) -> None:
        if not (isinstance(self.min, int) and self.min >= 0):
            raise ValueError(f"trial_length.min must be a non-negative int, got {self.min!r}")
        if not (isinstance(self.max, int) and self.max >= self.min):
            raise ValueError(
                f"trial_length.max must be an int >= min ({self.min}), got {self.max!r}"
            )


@dataclass
class DatasetCfg:
    seed: int
    alphabet: AlphabetCfg
    trial_length: TrialLengthCfg
    splits: dict[str, int | float]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError(f"dataset.seed must be int, got {self.seed!r}")
        if not self.splits:
            raise ValueError("dataset.splits must be non-empty")
        for name, value in self.splits.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"dataset.splits[{name!r}] must be int (count) or float in (0,1] "
                    f"(fraction), got {value!r}"
                )
            if isinstance(value, float) and not (0.0 < value <= 1.0):
                raise ValueError(
                    f"dataset.splits[{name!r}] float must be in (0,1], got {value!r}"
                )
            if isinstance(value, int) and value < 0:
                raise ValueError(
                    f"dataset.splits[{name!r}] int must be >= 0, got {value!r}"
                )

    def split_size(self, name: str, total: int | None = None) -> int:
        """Return the absolute sample count for a named split.

        If the stored value is already an int it is returned directly.
        If it is a float fraction, *total* must be provided; the result is
        ``round(fraction * total)``.  Passing no *total* for a fractional split
        raises ``TypeError`` rather than silently truncating to 0.
        """
        value = self.splits[name]
        if isinstance(value, float):
            if total is None:
                raise TypeError(
                    f"dataset.splits[{name!r}] is a fraction ({value!r}) — "
                    "provide a total sample count to resolve it"
                )
            return round(value * total)
        return value


# ----------------------------- symseq -----------------------------------------


@dataclass
class GeneratorCfg:
    type: str
    mode: AGMode | None = None        # AG-only; ignored for other types
    params: dict[str, Any] = field(default_factory=dict)
    trial_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"generator.type must be a non-empty string, got {self.type!r}")
        if self.mode is not None and not isinstance(self.mode, AGMode):
            # Allow string -> enum coercion at parse time.
            self.mode = AGMode(self.mode)
        # AG-mode default is "random"; non-AG mode must be None.
        if self.type == "ArtificialGrammar" and self.mode is None:
            self.mode = AGMode.RANDOM


@dataclass
class SymseqTaskEntry:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"symseq.tasks[*].name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"symseq.tasks[*].type must be a non-empty string, got {self.type!r}")


@dataclass
class SymseqStorageCfg:
    path: str
    format: str = "pickle"

    def __post_init__(self) -> None:
        if self.format != "pickle":
            raise ValueError(
                f"symseq.storage.format only supports 'pickle' for now, got {self.format!r}"
            )


@dataclass
class SymseqCfg:
    generator: GeneratorCfg
    seed: int | None = None
    tasks: list[SymseqTaskEntry] = field(default_factory=list)
    storage: SymseqStorageCfg | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for t in self.tasks:
            if t.name in seen:
                raise ValueError(f"symseq.tasks[*].name must be unique; duplicate {t.name!r}")
            seen.add(t.name)


# ----------------------------- seqbench ---------------------------------------


@dataclass
class SeqbenchStorageCfg:
    path: str
    force_rebuild: bool = False
    cache_key: str | None = None


@dataclass
class GapProfileCfg:
    start: int = 0
    duration: dict[str, Any] = field(default_factory=dict)
    add_nongramm_gap: bool = False


@dataclass
class CompositionCfg:
    combine_sequences: bool
    sample_length: int
    gap_profile: GapProfileCfg | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_length, int) or self.sample_length < 1:
            raise ValueError(
                f"composition.sample_length must be a positive int, got {self.sample_length!r}"
            )


@dataclass
class InputMappingCfg:
    base: str
    base_params: dict[str, Any] = field(default_factory=dict)
    transforms: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.base, str) or not self.base:
            raise ValueError(f"input_mapping.base must be a non-empty string, got {self.base!r}")
        for i, t in enumerate(self.transforms):
            if not isinstance(t, dict) or "name" not in t:
                raise ValueError(
                    f"input_mapping.transforms[{i}] must be a dict with a 'name' key, got {t!r}"
                )


@dataclass
class SeqbenchTaskCfg:
    source: TaskSource
    name: str | None = None      # source=symseq: must reference a symseq.tasks entry
    type: str | None = None      # source=seqbench: name from seqbench.tasks registry
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, TaskSource):
            self.source = TaskSource(self.source)
        if self.source == TaskSource.SYMSEQ:
            if not self.name:
                raise ValueError("seqbench.task with source='symseq' requires 'name'")
            if self.type:
                raise ValueError(
                    "seqbench.task with source='symseq' must NOT set 'type' "
                    "(use 'name' to point at a symseq.tasks entry)"
                )
        else:  # SEQBENCH
            if not self.type:
                raise ValueError("seqbench.task with source='seqbench' requires 'type'")
            if self.name:
                raise ValueError(
                    "seqbench.task with source='seqbench' must NOT set 'name' "
                    "(use 'type' to name a registered seqbench task)"
                )


@dataclass
class SeqbenchCfg:
    mode: str
    storage: SeqbenchStorageCfg
    composition: CompositionCfg
    input_mapping: InputMappingCfg
    task: SeqbenchTaskCfg
    seed: int | None = None
    prob_generator_type: str = "restricted"
    # Global grid resolution in seconds. Single source of truth shared by the
    # encoding (stimulus duration -> steps) and the gap profile (gap seconds ->
    # steps). See input_mapping.base_params.duration and gap_profile.duration.
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("file", "offline", "online"):
            raise ValueError(
                f"seqbench.mode must be 'file' | 'offline' | 'online', got {self.mode!r}"
            )
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError(f"seqbench.dt must be a positive number, got {self.dt!r}")


# ----------------------------- root -------------------------------------------


@dataclass
class RunConfig:
    dataset: DatasetCfg
    symseq: SymseqCfg | None = None
    seqbench: SeqbenchCfg | None = None

    def __post_init__(self) -> None:
        if self.symseq is None and self.seqbench is None:
            raise ValueError("At least one of 'symseq' or 'seqbench' must be present")
        # Cross-section consistency: source='symseq' requires a symseq.tasks entry.
        if self.symseq is not None and self.seqbench is not None:
            if self.seqbench.task.source == TaskSource.SYMSEQ:
                names = {t.name for t in self.symseq.tasks}
                if self.seqbench.task.name not in names:
                    raise ValueError(
                        f"seqbench.task.name={self.seqbench.task.name!r} does not match any "
                        f"symseq.tasks[*].name (available: {sorted(names) or 'none'})"
                    )
        if self.seqbench is not None and self.seqbench.task.source == TaskSource.SYMSEQ:
            if self.symseq is None:
                raise ValueError(
                    "seqbench.task.source='symseq' requires the top-level 'symseq' section"
                )

    @property
    def operating_mode(self) -> OperatingMode:
        if self.symseq is not None and self.seqbench is None:
            return OperatingMode.SYMSEQ_ONLY
        if self.symseq is not None and self.seqbench is not None:
            return OperatingMode.SYMSEQ_PLUS_SEQBENCH
        return OperatingMode.SEQBENCH_STANDALONE


# ----------------------------- parser -----------------------------------------


def load(source: str | Path | dict) -> RunConfig:
    """Load a YAML file (or pre-parsed dict) into a validated :class:`RunConfig`."""
    raw = _load_raw(source)
    return _parse_root(raw)


def _load_raw(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".yaml", ".yml"):
        raise ValueError(f"unsupported config extension {suffix!r}; use .yaml/.yml")
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"config file {path} is empty")
    return cfg


def _parse_root(raw: dict) -> RunConfig:
    _require_keys(raw, {"dataset"}, where="<root>")
    ds = _parse_dataset(raw["dataset"])
    symseq = _parse_symseq(raw["symseq"]) if "symseq" in raw else None
    seqbench = _parse_seqbench(raw["seqbench"]) if "seqbench" in raw else None
    return RunConfig(dataset=ds, symseq=symseq, seqbench=seqbench)


def _parse_dataset(raw: dict) -> DatasetCfg:
    _require_keys(raw, {"seed", "alphabet", "trial_length", "splits"}, where="dataset")
    return DatasetCfg(
        seed=raw["seed"],
        alphabet=AlphabetCfg(**raw["alphabet"]),
        trial_length=TrialLengthCfg(**raw["trial_length"]),
        splits=dict(raw["splits"]),
    )


def _parse_symseq(raw: dict) -> SymseqCfg:
    _require_keys(raw, {"generator"}, where="symseq")
    gen = GeneratorCfg(**raw["generator"])
    tasks = [SymseqTaskEntry(**t) for t in (raw.get("tasks") or [])]
    storage = SymseqStorageCfg(**raw["storage"]) if raw.get("storage") else None
    return SymseqCfg(
        generator=gen,
        seed=raw.get("seed"),
        tasks=tasks,
        storage=storage,
    )


def _parse_seqbench(raw: dict) -> SeqbenchCfg:
    _require_keys(
        raw,
        {"mode", "storage", "composition", "input_mapping", "task"},
        where="seqbench",
    )
    storage = SeqbenchStorageCfg(**raw["storage"])
    comp_raw = dict(raw["composition"])
    gp_raw = dict(comp_raw.pop("gap_profile", None) or {}) or None

    # Resolve the global grid resolution `dt`. Canonical home is seqbench.dt.
    # Back-compat: lift a legacy gap_profile.dt up to seqbench.dt (deprecated).
    dt = raw.get("dt")
    if gp_raw is not None and "dt" in gp_raw:
        legacy_dt = gp_raw.pop("dt")
        if dt is None:
            warnings.warn(
                "gap_profile.dt is deprecated; move it to the top-level seqbench.dt. "
                f"Using gap_profile.dt={legacy_dt!r} as seqbench.dt.",
                DeprecationWarning,
                stacklevel=2,
            )
            dt = legacy_dt
        else:
            warnings.warn(
                "Both seqbench.dt and the deprecated gap_profile.dt are set; "
                f"ignoring gap_profile.dt={legacy_dt!r} in favor of seqbench.dt={dt!r}.",
                DeprecationWarning,
                stacklevel=2,
            )

    composition = CompositionCfg(
        **comp_raw,
        gap_profile=GapProfileCfg(**gp_raw) if gp_raw else None,
    )
    input_mapping = InputMappingCfg(**raw["input_mapping"])
    task = SeqbenchTaskCfg(**raw["task"])
    return SeqbenchCfg(
        mode=raw["mode"],
        storage=storage,
        composition=composition,
        input_mapping=input_mapping,
        task=task,
        seed=raw.get("seed"),
        prob_generator_type=raw.get("prob_generator_type", "restricted"),
        dt=dt if dt is not None else SeqbenchCfg.dt,
    )


def _require_keys(d: dict, keys: set[str], *, where: str) -> None:
    missing = keys - set(d)
    if missing:
        raise ValueError(f"{where}: missing required keys {sorted(missing)}")
