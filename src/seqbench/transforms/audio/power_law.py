# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Power law transform for audio signal processing.
"""

import torch


class PowerLaw:
    """
    Apply power-law scaling: x -> x**gamma.
    Assumes input x is already normalized to [0,1].
    """

    def __init__(self, gamma=5.0):
        self.gamma = gamma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, self.gamma)
