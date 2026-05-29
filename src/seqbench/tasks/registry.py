# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Task registry — mirrors ``symseq.tasks.registry``.

Each task class registers itself via ``@register("Name")``; ``build(name, **params)``
instantiates a registered task.
"""

from __future__ import annotations

from typing import Type

from seqbench.tasks.base import Task


_REGISTRY: dict[str, Type[Task]] = {}


def register(name: str):
    """Class decorator that registers a Task under ``name``."""

    def _decorator(cls: Type[Task]) -> Type[Task]:
        if name in _REGISTRY:
            raise ValueError(
                f"Task {name!r} is already registered to {_REGISTRY[name].__name__}."
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(name: str, **params) -> Task:
    """Instantiate a registered Task by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**params)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
