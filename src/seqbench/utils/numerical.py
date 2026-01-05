# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Numerical utility functions for tensor operations and validation.
"""

import torch


def is_integer_multiple(a, b, tol=1e-8):
    q = a / b
    return abs(q - round(q)) < tol


def is_binary(tensor: torch.Tensor, threshold: float = 1e-6) -> bool:
    """
    Check if tensor contains only binary values (0 and 1).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to check
    threshold : float, optional
        Tolerance for floating point comparison, by default 1e-6

    Returns
    -------
    bool
        True if tensor contains only binary values
    """
    unique_vals = torch.unique(tensor)
    return len(unique_vals) <= 2 and torch.allclose(unique_vals, torch.round(unique_vals), atol=threshold)


def get_device_and_dtype(data):

    if isinstance(data, torch.Tensor):
        device = data.device
        dtype = data.dtype if data.dtype.is_floating_point else torch.float32
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype
