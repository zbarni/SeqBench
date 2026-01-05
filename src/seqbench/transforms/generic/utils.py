# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Generic transform utilities for device and dtype management.
"""

import torch


def get_device_and_dtype(data):

    if isinstance(data, torch.Tensor):
        device = data.device
        dtype = data.dtype if data.dtype.is_floating_point else torch.float32
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype
