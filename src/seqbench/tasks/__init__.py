# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Task definitions. Importing this package eagerly registers all built-in
tasks so :func:`seqbench.tasks.registry.build` works without per-class imports.

Naming mirrors ``symseq.tasks`` so a config can swap
``task.source: symseq`` ↔ ``task.source: seqbench`` for the same task name
(e.g. ``NStepPrediction``) without changes to params.
"""

from seqbench.tasks.base import Target, Task
from seqbench.tasks.classify import Classification, StateClassification
from seqbench.tasks.shift import NStepMemory, NStepPrediction

__all__ = [
    "Task",
    "Target",
    "NStepMemory",
    "NStepPrediction",
    "Classification",
    "StateClassification",
]
