# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Rate coding transform for converting continuous values to spike rates.
"""

import torch


class RateCoding:
    """
    Convert continuous-valued data into firing rate (Hz or arbitrary units).

    This transform maps arbitrary numeric input (audio features, pixels,
    embeddings...) into non-negative firing rates suitable for neuromorphic
    encoding. Typically used before PoissonEncoder or other spike encoders.

    Parameters
    ----------
    max_rate : float
        Upper bound for the output rate (e.g., 100–500 Hz typical).
    normalize : bool
        If True, input is normalized to [0, 1] per-sample before scaling.
        This is useful when inputs have variable range.
    clamp : bool
        If True, negative values are clamped to 0 after scaling.
    eps : float
        Small constant to avoid division-by-zero in normalization.

    Examples
    --------
    >>> rate = RateCoding(max_rate=200.0, normalize=True)
    >>> x = torch.randn(1, 128, 100)   # MFCC-like features
    >>> rates = rate(x)                # output in [0, 200]

    >>> # Without normalization:
    >>> rate = RateCoding(max_rate=100.0)
    >>> rates = rate(torch.sigmoid(x))
    """

    def __init__(
        self,
        max_rate: float = 200.0,
        normalize: bool = False,
        clamp: bool = True,
        eps: float = 1e-8,
    ):
        self.max_rate = float(max_rate)
        self.normalize = normalize
        self.clamp = clamp
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert arbitrary input tensor into non-negative rate values.

        Expected input: arbitrary float tensor (any shape)
        Output: same shape, values in [0, max_rate] or unclamped if clamp=False
        """
        if not torch.is_floating_point(x):
            x = x.float()

        # # Optional normalization to [0,1] per sample or batch element
        # if self.normalize:
        #     # Compute per-sample min/max across all non-batch dims
        #     # Works for shapes like (B, C, T, H, W) or (C, T) or (T,)
        #     dims = tuple(range(1, x.ndim))
        #     x_min = x.amin(dim=dims, keepdim=True)
        #     x_max = x.amax(dim=dims, keepdim=True)

        #     # Scale to [0,1]
        #     x = (x - x_min) / (x_max - x_min + self.eps)
        if self.normalize:
            x_min, x_max = x.min(), x.max()
            if x_max <= x_min:
                return torch.zeros_like(x)
            rates = (x - x_min) / (x_max - x_min)

        # Scale to rate
        rates = x * self.max_rate

        # Clamp negative or overly large values
        if self.clamp:
            rates = torch.clamp(rates, min=0.0, max=self.max_rate)

        return rates

    def __repr__(self):
        return str(
            (
                f"RateCoding(max_rate={self.max_rate}, ",
                f"normalize={self.normalize}, ",
                f"clamp={self.clamp})",
            )
        )
