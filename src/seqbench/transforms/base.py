# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Base transform protocol for defining transform interfaces.
"""

from typing import Protocol, Any


class Transform(Protocol):
    def __call__(self, x: Any) -> Any: ...
