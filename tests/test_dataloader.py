# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Tests for the SeqBench dataset creation and DataLoader functionality.
"""

import pytest
import numpy as np
import random
import torch
import os

from torch.utils.data import DataLoader

try:
    from seqbench.sources import build_symseq_source
    import symseq  # noqa: F401  # verify availability
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench.seq_dataset import SeqDataset, PadSequence
from seqbench.utils.config import Config
from seqbench.utils import prepare_config, get_config_hash
from seqbench.dataset import create_base_dataset_from_config
from seqbench.transforms import compose_transforms_from_config


@pytest.fixture(scope="class")
def dataset_setup():
    """Set up dataset for testing."""
    config_path = os.path.join(os.path.dirname(__file__), 'onehot_raw.yaml')
    
    if not os.path.exists(config_path):
        pytest.skip(f"Config file {config_path} not found")
    
    if not HAS_SYMSEQ:
        pytest.skip("symseq not available")
    
    # Load configuration
    args = {"config": config_path}
    config = Config.parse_config_from_args(args)
    
    # Prepare seqbench config
    seqbench_config = prepare_config(config["seqbench"])
    seed = seqbench_config["seed"]
    seqbench_config["do_classify"] = True
    seqbench_config["dataset_size"] = min(seqbench_config["data_generation"]["train_size"], 20)
    seqbench_config["config_file_path"] = config_path
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Build symseq trial source
    source = build_symseq_source(config["symseq"]["generator"])

    # Create base dataset
    kwargs = {}
    if seqbench_config["input_mapping"]["base"] == "one_hot":
        kwargs["alphabet_size"] = len(source.alphabet)

    base_dataset = create_base_dataset_from_config(seqbench_config, "train", **kwargs)
    
    # Compose transforms
    transforms = compose_transforms_from_config(seqbench_config)
    
    # Get dataset root
    config_hash = get_config_hash(config)
    dataset_root = seqbench_config["data_generation"]["output_dir"]
    dataset_root = f"{dataset_root}/{config_hash}"
    
    # Create SeqDataset
    dataset = SeqDataset(
        config=seqbench_config,
        generator=source,
        base_dataset=base_dataset,
        is_train=True,
        pad_index=-1,
        dataset_root=dataset_root,
        transform=transforms,
    )
    
    return dataset, seqbench_config


class TestDataset:
    """Tests for dataset creation and basic functionality."""

    def test_dataset_creation(self, dataset_setup):
        """Test that dataset is created successfully."""
        dataset, seqbench_config = dataset_setup
        
        assert dataset is not None
        assert len(dataset) > 0
        assert len(dataset) == seqbench_config["dataset_size"]

    def test_dataset_getitem(self, dataset_setup):
        """Test that __getitem__ returns a valid sample."""
        dataset, seqbench_config = dataset_setup
        
        sample = dataset[0]
        assert sample is not None
        assert len(sample) == 4  # data, target, class_seq, gap_mask
        
        data, target, class_seq, gap_mask = sample
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape[0] == target.shape[0]


class TestDataLoader:
    """Tests for DataLoader integration."""

    def test_dataloader_iteration(self, dataset_setup):
        """Test that DataLoader can iterate over batches."""
        dataset, seqbench_config = dataset_setup
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=PadSequence(do_classify=seqbench_config["do_classify"], pad_index=-1),
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        
        assert isinstance(batch, dict)
        assert 'data' in batch
        assert 'labels' in batch
        assert 'mask' in batch
        assert 'lens' in batch

    def test_dataloader_batch_shapes(self, dataset_setup):
        """Test that DataLoader batches have correct shapes."""
        dataset, seqbench_config = dataset_setup
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=PadSequence(do_classify=seqbench_config["do_classify"], pad_index=-1),
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        
        assert batch['data'].dim() == 3  # (batch, time, features)
        assert batch['data'].shape[0] <= 4
        assert batch['labels'].shape[0] == batch['data'].shape[0]
        assert batch['labels'].shape[1] == batch['data'].shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
