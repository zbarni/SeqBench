# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence-level classification: a single label per sample.

Covers two regimes:
 - seqbench-standalone with a base dataset that carries its own labels
   (e.g. GSC keyword spotting) — ``label_source="base"``.
 - probes over the symbolic class_seq (e.g. "what was the first/last
   symbol?") — ``label_source="first" | "last"``.
"""

from __future__ import annotations

from typing import Literal

from seqbench.seq_utils.generator import GeneratorSample
from seqbench.tasks.base import Target, Task
from seqbench.tasks.registry import register


LabelSource = Literal["base", "first", "last"]


@register("Classification")
class Classification(Task):
    """One label per sample.

    Parameters
    ----------
    label_source : {"base", "first", "last"}
        - ``"base"``: read the label attached by the base dataset
          (``sample.label`` — must be populated by the input_mapping pipeline).
        - ``"first"``: ``class_seq[0]``.
        - ``"last"``:  ``class_seq[-1]``.
    """

    def __init__(self, label_source: LabelSource = "base"):
        if label_source not in ("base", "first", "last"):
            raise ValueError(
                f"label_source must be one of 'base'|'first'|'last', got {label_source!r}"
            )
        self.label_source = label_source
        self.name = f"classification_{label_source}"

    def __call__(self, sample: GeneratorSample) -> Target:
        if self.label_source == "base":
            label = getattr(sample, "label", None)
            if label is None:
                raise ValueError(
                    "Classification(label_source='base') requires sample.label "
                    "to be set by the input_mapping (base dataset label)."
                )
            return Target(values=label, mask=None, kind="per_trial")

        seq = list(sample.class_seq)
        if not seq:
            return Target(values=None, mask=None, kind="per_trial")
        value = seq[0] if self.label_source == "first" else seq[-1]
        return Target(values=value, mask=None, kind="per_trial")
