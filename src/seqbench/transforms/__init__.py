# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Transform Module for SeqBench

This module provides data transformation utilities and composition functionality.
"""

import pkgutil
import importlib
import inspect
import ast
from pathlib import Path

from seqbench.transforms.compose import Compose
from seqbench.transforms.functional import FunctionalTransform

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = []


def _auto_import_classes(package_name: str, package_path: Path):
    """
    Recursively import all classes from the given package and subpackages.
    """

    # Ensure package_path is a Path object
    package_path = Path(package_path)

    for finder, module_name, ispkg in pkgutil.walk_packages([str(package_path)], prefix=package_name + "."):
        # Skip private modules
        if any(part.startswith("_") for part in module_name.split(".")):
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue  # or log

        # Import all classes defined in this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name:
                globals()[name] = obj
                __all__.append(name)


# Run for current package
_package_name = __name__
_package_path = Path(__file__).parent

_auto_import_classes(_package_name, _package_path)


def _normalize_params(params):
    """Normalize YAML params: 'None' -> None, '(a,b)' -> tuple, 'torch.X' -> torch.X"""
    normalized = {}
    for k, v in params.items():
        if isinstance(v, str):
            if v.lower() == 'none':
                v = None
            elif v.startswith('torch.') and TORCH_AVAILABLE:
                v = getattr(torch, v.split('.', 1)[1], v)
            elif v.startswith('(') and v.endswith(')'):
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
        normalized[k] = v
    return normalized


def compose_transforms_from_config(config_or_input_mapping):
    """
    Create a Compose object from the transforms defined in the config.
    Supports dotted paths (e.g., "tonic.transforms.ToFrame") and functions
    (wrapped in FunctionalTransform).

    Accepts either an :class:`~seqbench.config.InputMappingCfg` (new path,
    list-of-dicts format) or a legacy ``Config`` object with an
    ``input_mapping`` key (old path, name→params dict format).
    """
    # New path: InputMappingCfg passed directly (has .transforms as a dataclass field).
    if hasattr(config_or_input_mapping, 'transforms'):
        transforms_spec = config_or_input_mapping.transforms  # list[dict]
        if not transforms_spec:
            return None
        # [{name: "ExpandDim", axis: 1, ...}, ...] → [(name, {axis: 1, ...}), ...]
        items = [
            (t['name'], {k: v for k, v in t.items() if k != 'name'})
            for t in transforms_spec
        ]
    else:
        # Old path: full Config with input_mapping key.
        inp_map = config_or_input_mapping["input_mapping"]
        if "transforms" not in inp_map or inp_map["transforms"] is None:
            return None
        transforms_dict = inp_map["transforms"]
        if not transforms_dict:
            return None
        items = list(transforms_dict.items())

    transforms = []
    for name, params in items:
        params = _normalize_params(params or {})

        try:
            # Handle dotted paths (e.g., "tonic.transforms.ToFrame")
            if "." in name:
                parts = name.split(".")
                module = importlib.import_module(".".join(parts[:-1]))
                obj = getattr(module, parts[-1])
            else:
                obj = globals()[name]

            # Wrap functions, instantiate classes
            if inspect.isclass(obj):
                t = obj(**params)
            elif callable(obj):
                t = FunctionalTransform(obj, **params)
            else:
                raise ValueError(f"{name} is not callable")

        except Exception as e:
            raise RuntimeError(f"Could not create transform {name} with params {params}") from e

        transforms.append(t)

    return Compose(transforms)
