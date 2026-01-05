# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Transform composition utilities for chaining multiple transforms together.
"""

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ApplyToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, sample):
        sample = dict(sample)
        sample[self.key] = self.transform(sample[self.key])
        return sample
