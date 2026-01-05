# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Functional transform wrapper for converting pure functions into transform objects.
"""

from functools import partial
from typing import Callable, Any


class FunctionalTransform:
    """
    Wraps a pure function (e.g., from torchaudio.functional or custom) into a callable
    transform that can be used in Compose pipelines.

    Example usage:
        from torchaudio.functional import amplitude_to_DB
        transform = FunctionalTransform(amplitude_to_DB, multiplier=20, amin=1e-10)
        output = transform(waveform)

    Attributes:
        func: Callable function to wrap.
        kwargs: Keyword arguments to bind to the function.
    """

    def __init__(self, func: Callable, **kwargs):
        if not callable(func):
            raise ValueError(f"func must be callable, got {type(func)}")
        self.func = partial(func, **kwargs)

    def __call__(self, x: Any) -> Any:
        """
        Apply the wrapped function to input x.
        """
        return self.func(x)

    def __repr__(self):
        kwarg_str = (
            ", ".join(f"{k}={v}" for k, v in self.func.keywords.items()) if hasattr(self.func, "keywords") else ""
        )
        return f"{self.__class__.__name__}({self.func.func.__name__}({kwarg_str}))"
