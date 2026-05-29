# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Task ABC + Target — SeqBench's task layer, mirroring ``symseq.tasks.base``.

A SeqBench Task derives a Target from a ``GeneratorSample`` (the concatenated,
encoded sample produced by ``seqbench.seq_utils.generator.GeneratorSample``).
The symbolic, per-token logic operates on ``sample.class_seq`` (numeric class
ids); embedding-derived tasks may also reach into the encoded tensors.

The Target dataclass intentionally matches ``symseq.tasks.base.Target`` in
shape so that downstream code can treat both registries uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from seqbench.seq_utils.generator import GeneratorSample


@dataclass
class Target:
    values: list[Any] | Any
    mask: list[bool] | None
    kind: Literal["per_token", "per_trial"]


class Task(ABC):
    """Abstract base for SeqBench tasks."""

    name: str

    @abstractmethod
    def __call__(self, sample: GeneratorSample) -> Target: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
