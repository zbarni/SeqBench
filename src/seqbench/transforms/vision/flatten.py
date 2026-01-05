# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Flatten transform for reshaping vision data tensors.
"""

class Flatten:
    def __init__(self, start_dim):
        self.start_dim = start_dim

    def __call__(self, img):
        return img.flatten(start_dim=self.start_dim)