# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

# Backward-compatibility shim — import from seqbench.config instead.
from seqbench.config import (  # noqa: F401
    load,
    RunConfig,
    DatasetCfg,
    AlphabetCfg,
    TrialLengthCfg,
    GeneratorCfg,
    SymseqTaskEntry,
    SymseqStorageCfg,
    SymseqCfg,
    SeqbenchStorageCfg,
    TimeGridCfg,
    GapProfileCfg,
    CompositionCfg,
    InputMappingCfg,
    SeqbenchTaskCfg,
    SeqbenchCfg,
    OperatingMode,
    AGMode,
    TaskSource,
)
