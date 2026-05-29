# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence Dataset Module for SeqBench

This module provides PyTorch dataset classes and utilities for handling sequential
data in the SeqBench framework. It includes support for various sequence embeddings,
temporal processing, and various sequence tasks.
"""

# Standard library imports
import os
import logging
from math import ceil
from dataclasses import dataclass
from typing import Optional, Any

# Third-party imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import tqdm

# Local imports
from seqbench.utils import get_config_hash
from seqbench.dataset import create_base_dataset_from_config
from seqbench.seq_utils.generator import SequenceGenerator
from seqbench.dataset_generator import DatasetGenerator, RestrictedTargetProbGenerator

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class _GapProfile:
    """Normalised gap profile used internally by SeqDataset.

    Produced from either a :class:`~seqbench.config.GapProfileCfg` (new path)
    or the legacy Config dict (old path) so that _add_gap_content does not
    need to know which config system it came from.
    """

    start: int = 0
    gap_prob: float = 1.0
    duration: Any = None  # dict {dist: callable, params: dict} or scalar/None
    add_nongramm_gap: bool = False
    dt: float = 0.1
    noise: Optional[float] = None


@dataclass
class Sample:
    """
    Data structure representing a single sequence sample.

    Attributes:
        class_seq: Sequence of class indices (int array)
        state_seq: Sequence of state symbols (str array)
        target_seq: Target sequence for prediction (int array)
        target_probs: Target probabilities for each timestep (float array)
        length: Length of the sequence (int)
    """

    class_seq: np.ndarray  # int
    state_seq: np.ndarray  # str
    target_seq: np.ndarray  # int
    target_probs: np.ndarray  # float
    length: int


class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequential data processing in SeqBench.

    This dataset class handles various sequence encodings, temporal processing,
    and classification tasks. It supports both training and evaluation modes
    with configurable padding, transformations, and data loading strategies.

    Key Features:
    - Support for multiple input encodings (one-hot, SHD, SSC, etc.)
    - Temporal sequence processing with configurable timesteps
    - Classification and regression target generation
    - Configurable padding and data augmentation
    - Efficient data loading with optional caching

    Accepts either a :class:`~seqbench.config.RunConfig` (new path — pass
    ``dataset_size``, ``config_file_path``, and ``do_classify`` explicitly) or
    the legacy ``Config`` wrapper (old path — those values are read from the
    config object for backward compatibility with callers not yet migrated).

    Attributes:
        base_dataset: Base dataset providing raw sequence data
        is_train: Whether this is a training dataset
        timestep: Temporal resolution for sequence processing
        do_classify: Whether to perform classification tasks
        pad_index: Index used for sequence padding
        dataset_size: Total number of samples in the dataset
    """

    def __init__(
        self,
        config,
        generator: Any,
        base_dataset: Any,
        is_train: bool,
        dataset_size: Optional[int] = None,
        config_file_path: Optional[str] = None,
        do_classify: Optional[bool] = None,
        pad_index: int = -1,
        not_temporal: bool = False,
        base_duration: Optional[int] = None,
        load_to_tensor: Optional[callable] = None,
        transform: Optional[callable] = None,
        dataset_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the SeqDataset.

        Args:
            config: :class:`~seqbench.config.RunConfig` (new path) or legacy
                ``Config`` wrapper (old path).
            generator: Sequence generator / trial source for creating sequences.
            base_dataset: Base dataset providing raw data samples.
            is_train: Whether this is a training dataset.
            dataset_size: Number of samples to use.  Required when passing a
                ``RunConfig``; read from the config object on the legacy path.
            config_file_path: Path to the YAML config file written into the
                generated dataset directory.  Required on the RunConfig path.
            do_classify: Whether to use classification targets.  Required on
                the RunConfig path; read from the config object on the legacy
                path.  Will later be derived from the tasks module.
            pad_index: Index to use for padding sequences (default: -1).
            not_temporal: If True, treat data as non-temporal (default: False).
            base_duration: Base duration of samples in timesteps
                (auto-detected if None).
            load_to_tensor: Optional function to convert data to tensors.
            transform: Optional transform to apply to data samples.
            dataset_root: Root directory for dataset storage (default: None).
        """
        from seqbench.config import RunConfig

        if isinstance(config, RunConfig):
            self._init_from_run_cfg(config, dataset_size, config_file_path, do_classify)
        else:
            self._init_from_legacy_config(
                config, dataset_size, config_file_path, do_classify
            )

        self.base_dataset = base_dataset
        self.is_train = is_train
        self.transform = transform
        self.generator = generator
        self.dataset_root = dataset_root
        self.pad_index = pad_index
        self.not_temporal = not_temporal  # TODO potentially remove and set dynamically

        if base_duration:
            self.base_duration = base_duration
        else:
            self.base_duration = self.base_dataset[0][0].shape[0]

        if load_to_tensor:
            self.load_to_tensor = load_to_tensor
        else:
            self.load_to_tensor = lambda x: x

        if self._seed is None:
            self.rng = np.random.default_rng()
            logger.warning("Results will not be reproducible!")
        else:
            self.rng = np.random.default_rng(seed=self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._seed)

        base_sample = self.load_to_tensor(self.base_dataset[0][0])
        if self.not_temporal:
            base_sample = base_sample.unsqueeze(0)
        if self.transform:
            base_sample = self.transform(base_sample)
        self.base_shape = self.__get_dynamic_samples_dimensions(base_sample)

        # We create the target_prob_generator even for classification
        # to get the output_dim from the transitions
        if self.prob_generator_type == "restricted":
            self.target_prob_generator = RestrictedTargetProbGenerator()
        else:
            raise ValueError(
                f"Did not recognize target_prob_generator {self.prob_generator_type}!"
            )

        if self.read_from_file:
            if self.__dataset_folder_exists() and not self.regenerate:
                self.__prepare_dataset_from_file()
            else:
                print(
                    f"SeqBench: Did not find dataset in {self.dataset_root}. Creating it first!"
                )
                self.__create_dataset()
                print(f"SeqBench: Created new dataset in {self.dataset_root}!")
                self.__prepare_dataset_from_file()
        else:
            print("SeqBench: Generating dataset on the fly!")
            self.gs = self.__create_sequence_generator()
            self.target_prob_generator.read_transitions_from_generator(self.gs)

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _init_from_run_cfg(self, run_cfg, dataset_size, config_file_path, do_classify):
        """Extract all config-derived values from a :class:`~seqbench.config.RunConfig`."""
        from seqbench.utils import parse_gap_duration

        assert (
            dataset_size is not None
        ), "dataset_size must be provided when using RunConfig"
        assert (
            do_classify is not None
        ), "do_classify must be provided when using RunConfig"

        self._seed = run_cfg.dataset.seed
        self.dataset_size = dataset_size
        # TODO handle timestamp properly
        self.timestep = run_cfg.seqbench.input_mapping.base_params.get("timestep", None)
        self.read_from_file = (
            run_cfg.seqbench.mode == "file"
        )  # TODO self.config['read_from_file']
        self.prob_generator_type = run_cfg.seqbench.prob_generator_type
        self.do_classify = do_classify
        self.regenerate = run_cfg.seqbench.storage.force_rebuild
        self._config_file_path = config_file_path
        self._input_mapping_base = run_cfg.seqbench.input_mapping.base
        self._global_noise = 0

        # Sequence generator params
        self._seq_len_min = run_cfg.dataset.trial_length.min
        self._seq_len_max = run_cfg.dataset.trial_length.max
        self._combine_sequences = run_cfg.seqbench.composition.combine_sequences
        self._combined_seq_len = run_cfg.seqbench.composition.sample_length

        # Normalise gap profile
        gp_cfg = run_cfg.seqbench.composition.gap_profile
        if gp_cfg is not None:
            duration = dict(gp_cfg.duration)
            if "dist" in duration and isinstance(duration["dist"], str):
                duration["dist"] = parse_gap_duration(duration["dist"])
            self._gap_profile = _GapProfile(
                start=gp_cfg.start,
                gap_prob=1.0,  # not in new schema; always insert gaps
                duration=duration,
                add_nongramm_gap=gp_cfg.add_nongramm_gap,
                dt=gp_cfg.dt,
                noise=None,  # not in new schema
            )
        else:
            self._gap_profile = None

    def _init_from_legacy_config(
        self, config, dataset_size, config_file_path, do_classify
    ):
        """Extract all config-derived values from a legacy ``Config`` object."""
        self._seed = config["seed"]
        self.dataset_size = (
            dataset_size if dataset_size is not None else config["dataset_size"]
        )
        # TODO handle timestamp properly
        self.timestep = config["input_mapping"].get("timestep", None)
        self.read_from_file = (
            config["mode"] == "file"
        )  # TODO self.config['read_from_file']
        self.prob_generator_type = config["prob_generator_type"]
        self.do_classify = (
            do_classify if do_classify is not None else config["do_classify"]
        )
        self.regenerate = config["data_generation"]["regenerate"]
        self._config_file_path = (
            config_file_path
            if config_file_path is not None
            else config["config_file_path"]
        )
        self._input_mapping_base = config["input_mapping"]["base"]
        self._global_noise = config["noise", 0]

        # Sequence generator params
        data_gen = config["data_generation"]
        self._seq_len_min = data_gen["seq_len_min"]
        self._seq_len_max = data_gen["seq_len_max"]
        self._combine_sequences = data_gen["combine_sequences"]
        self._combined_seq_len = data_gen["combined_seq_length"]

        # Normalise gap profile
        if "gap_profile" in config and config["gap_profile"] is not None:
            gp = config["gap_profile"]
            raw_duration = gp["duration"]
            # Legacy: duration may be a Config wrapping {dist: callable, params: dict}
            # or a scalar.  Keep as-is — _add_gap_content handles both.
            self._gap_profile = _GapProfile(
                start=gp["start", 0],
                gap_prob=gp["gap_prob", 1],
                duration=raw_duration,
                add_nongramm_gap=gp["add_nongramm_gap", False],
                dt=gp["dt"],
                noise=gp["noise", None],
            )
        else:
            self._gap_profile = None

    # ------------------------------------------------------------------
    # Private dataset methods
    # ------------------------------------------------------------------

    def __dataset_folder_exists(self):
        """
        Check if the dataset folder exists and contains required files.

        Returns:
            bool: True if dataset folder exists with all required files, False otherwise.
        """
        if self.dataset_root is None:
            return False
        if not os.path.isdir(self.dataset_root):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, "config.yaml")):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, "transitions")):
            return False
        if self.is_train and not os.path.isfile(
            os.path.join(self.dataset_root, "train")
        ):
            return False
        if not self.is_train and not os.path.isfile(
            os.path.join(self.dataset_root, "test")
        ):
            return False
        return True

    def __create_dataset(self):
        """
        Create a new dataset by generating sequences and saving them to disk.

        This method creates a DatasetGenerator and generates the dataset files
        (train or test) based on the configuration.
        """
        print(f"SeqBench: Generating dataset with config {self._config_file_path}!")
        self.gs = self.__create_sequence_generator()
        dataset_generator = DatasetGenerator(
            self.gs,
            dataset_size=self.dataset_size,
            output_dir=self.dataset_root,
            config_file_path=self._config_file_path,
            generate_train=self.is_train,
            generate_test=(not self.is_train),
        )
        dataset_generator.generate()

    def __prepare_dataset_from_file(self):
        """
        Prepare the dataset by loading samples from pre-generated files.

        This method reads transition probabilities from file and buffers all
        samples into memory for fast access.
        """
        print(f"SeqBench: Reading dataset from {self.dataset_root}!")
        self.target_prob_generator.read_transitions_from_file(
            os.path.join(self.dataset_root, "transitions")
        )
        self.buffered_samples = self.__buffer_samples_from_file()

    def __create_sequence_generator(self):
        """
        Create a sequence generator from the instance configuration values.

        Returns:
            SequenceGenerator: Configured sequence generator instance.
        """
        return SequenceGenerator(
            self.generator,
            seq_len_min=self._seq_len_min,
            seq_len_max=self._seq_len_max,
            combine_sequences=self._combine_sequences,
            combined_seq_len=self._combined_seq_len,
            seed=self._seed,
        )

    @property
    def output_dim(self):
        """
        Get the output dimension for the dataset.

        For classification tasks, returns the number of unreduced states.
        For prediction tasks, returns the number of reduced states.

        Returns:
            int: Output dimension (number of classes or states).
        """
        if self.do_classify:
            return self.target_prob_generator.num_unreduced_states
        else:
            return self.target_prob_generator.num_reduced_states

    def __get_samples_dimensions(self):
        """
        Get the dimensions of samples from the base dataset.

        Returns:
            tuple: Sample dimensions (excluding temporal dimension if not_temporal is False).
        """
        data = self.base_dataset[0][0]
        shape = data.shape
        if self.not_temporal:
            return shape
        else:
            return shape[1:]

    def __get_dynamic_samples_dimensions(self, data):
        """
        Get the dimensions of a data sample dynamically.

        This method is used when sample dimensions may change after transformations.

        Args:
            data: Data tensor to get dimensions from.

        Returns:
            tuple: Sample dimensions (excluding temporal dimension if not_temporal is False).
        """
        shape = data.shape
        if self.not_temporal:
            return shape
        else:
            return shape[1:]

    def _add_gap_content(self, seq_element, state_idx, reshape):
        """
        Add gap content between sequence elements based on gap_profile configuration.

        Args:
            seq_element: Current position in the sequence.
            state_idx: Current state index.
            reshape: Function to reshape data samples.

        Returns:
            List of tensors representing the gap items.
        """
        gap_items = []
        gp = self._gap_profile

        if seq_element < gp.start:
            delay = 0
        else:
            if self.rng.random() > gp.gap_prob:
                delay = 0
            elif isinstance(gp.duration, dict) and "dist" in gp.duration:
                delay = np.round(
                    gp.duration["dist"](**gp.duration["params"]),
                    decimals=1,
                )
            else:
                delay = gp.duration

        if gp.add_nongramm_gap:
            preds_idx = np.where(
                self.target_prob_generator.transition_probs[state_idx] == 0
            )[0]
            delay_dur = 0
            n_delays = round(delay / (self.base_duration * self.timestep))
            if len(preds_idx) != 0:
                delays = []
                for i in range(n_delays):
                    idx = self.rng.choice(preds_idx)

                    if self._input_mapping_base == "one_hot":
                        delay_idx = self.base_dataset.class_dict[idx]
                    else:
                        delay_idx = int(
                            self.rng.choice(self.base_dataset.class_dict[idx])
                        )

                    delay_content = self.base_dataset[delay_idx][0]
                    delay_content = reshape(delay_content)
                    delays.append(delay_content)
                    delay_dur += delay_content.shape[0]

                gap_items.extend(delays)
            else:
                delay_content = torch.zeros([delay_dur] + list(self.base_shape))
                gap_items.append(delay_content)
        else:
            delay_dur = ceil(delay / gp.dt)
            delay_content = torch.zeros([delay_dur] + list(self.base_shape))
            gap_items.append(delay_content)

        if gp.noise is not None and not gp.add_nongramm_gap:
            for i, gap_item in enumerate(gap_items):
                gap_item = gap_item.float()
                rand_noise = torch.rand_like(gap_item) * self._global_noise
                gap_items[i] = gap_item + rand_noise

        return gap_items

    def __buffer_samples_from_file(self):
        """
        Load and buffer all samples from pre-generated dataset files.

        Returns:
            list: List of Sample objects loaded from the dataset file.
        """
        if self.is_train:
            data_path = os.path.join(self.dataset_root, "train")
        else:
            data_path = os.path.join(self.dataset_root, "test")

        buffered_samples = []
        pbar = tqdm.tqdm(total=self.dataset_size, desc="Buffering samples")
        with open(data_path, "r") as file:
            for line in file:
                line = line.replace("\n", "").strip()
                gensample = DatasetGenerator.create_gensample_from_str(line)
                sample = self.gensample_to_sample(gensample)
                buffered_samples.append(sample)
                pbar.update(1)
        pbar.close()
        return buffered_samples

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Returns an element of SeqBench. Since we do not need transition probabilities in case of
        classification, a returned element differs between classification and prediction.
        However, the only difference is that no target probabilities are provided
        (and the content of target_seq of course).
        Prediction: [input_seq, target_seq, class_seq, target_prob_seq]
        Classification: [input_seq, target_seq, class_seq], where:
        - input_seq has shape (T, J)
        - target_seq has shape (T, 1)
        - class_seq has shape (T, 1)
        - target_prob_seq has shape (T, C) (C is the number of ambiguous states)
        Note: class_seq is not needed and only provided for debugging.
        """
        assert not (torch.is_tensor(idx)), "idx needs to be an integer"

        sample = self.__get_sample(idx)

        if self.not_temporal:
            reshape = lambda x: self.load_to_tensor(x).unsqueeze(0)
        else:
            reshape = lambda x: self.load_to_tensor(x)

        if self.do_classify:
            # This is the place holder that should not be used in that context
            target_probs_or_placeholder = sample.class_seq
        else:
            target_probs_or_placeholder = sample.target_probs

        data = []
        target = []
        gap_mask = []
        target_probs = []
        timestamp = 0
        seq_element = 0

        for data_idx, target_idx, state_idx, target_prob in zip(
            sample.class_seq,
            sample.target_seq,
            sample.state_seq,
            target_probs_or_placeholder,
        ):

            if self._input_mapping_base == "one_hot":
                data_idx = self.base_dataset.class_dict[data_idx]
            else:
                data_idx = int(
                    self.rng.choice(self.base_dataset.class_dict[data_idx])
                )  # 1 2 3 0 -> 283 12002 1233 1737

            cur_data_sample = self.base_dataset[data_idx][0]
            cur_data_sample = reshape(cur_data_sample)

            # ---> apply any transforms here
            # TODO BZ: add noise transforms to sample!!
            if self.transform:
                # cur_data_sample_tmp = self.transform(cur_data_sample)
                # print("Sample shape before transform:", cur_data_sample.shape)
                # print("Sample shape after transform:", cur_data_sample_tmp.shape)
                # cur_data_sample = cur_data_sample_tmp
                cur_data_sample = self.transform(cur_data_sample)
                self.base_shape = self.__get_dynamic_samples_dimensions(cur_data_sample)

            data.append(cur_data_sample)

            if self._gap_profile is not None:
                gap_items = self._add_gap_content(seq_element, state_idx, reshape)
                data.extend(gap_items)
                delay_dur = (
                    sum(item.shape[0] for item in gap_items)
                    if len(gap_items) > 0
                    else 0
                )
            else:
                delay_dur = 0

            # ================================================================================

            data_t = cur_data_sample.shape[0]

            # TODO @BZ @Younes move logic to new *tasks* module
            target.append(torch.ones(data_t + delay_dur) * target_idx)

            if not self.do_classify:
                target_prob = torch.tensor(target_prob).unsqueeze(0)
                target_probs.append(target_prob.repeat(data_t + delay_dur, 1))

            if self.do_classify:
                gap_mask.append(
                    torch.cat(
                        [torch.zeros(timestamp + data_t - 1), torch.ones(delay_dur + 1)]
                    )
                )
            else:
                gap_mask.append(torch.zeros(data_t - 1))
                gap_mask.append(torch.ones(delay_dur + 1))

            timestamp = timestamp + data_t + delay_dur
            seq_element += 1

        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)

        if not self.do_classify:
            target_probs = torch.cat(target_probs, dim=0)

        assert data.shape[0] == target.shape[0]

        if self.do_classify:
            gap_mask = torch.nn.utils.rnn.pad_sequence(gap_mask, batch_first=True)
        else:
            gap_mask = torch.cat(gap_mask, dim=0)

        if self.do_classify:
            assert data.shape[0] == target.shape[0]
            sample = [data, target, sample.class_seq, gap_mask]
        else:
            assert data.shape[0] == target.shape[0] == target_probs.shape[0]
            class_seq = [
                self.target_prob_generator.unred_state_to_id(s)
                for s in sample.state_seq
            ]
            sample = [data, target, class_seq, target_probs, gap_mask]
        return sample

    def __get_sample(self, idx):
        """
        Get a Sample object for the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Sample: Sample object containing class_seq, state_seq, target_seq, etc.
        """
        if self.read_from_file:
            return self.__get_sample_from_buffer(idx)
        else:
            gensample = self.gs.generate(idx, compute_length=False)
            return self.gensample_to_sample(gensample)

    def __get_sample_from_buffer(self, idx):
        """
        Get a sample from the buffered samples.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Sample: Sample object from the buffer.
        """
        return self.buffered_samples[idx]

    def gensample_to_sample(self, gensample):
        """
        Convert a GeneratorSample to a Sample object.

        Args:
            gensample: GeneratorSample object from the sequence generator.

        Returns:
            Sample: Sample object with target sequences and probabilities.
        """
        target_seq = self.create_target_for_gensample(gensample, self.do_classify)
        if self.do_classify:
            target_probs = None
        else:
            target_probs = self.target_prob_generator(gensample.state_seq)
        return Sample(
            gensample.class_seq,
            gensample.state_seq,
            target_seq,
            target_probs,
            gensample.length,
        )

    def create_target_for_gensample(self, gensample, do_classify):
        """
        Create target sequence for a generator sample.

        For classification tasks, creates targets from unreduced state IDs.
        For prediction tasks, creates targets from class sequences with special
        handling for end-of-sequence tokens.

        Args:
            gensample: GeneratorSample object.
            do_classify: Whether this is for classification (True) or prediction (False).

        Returns:
            np.ndarray: Target sequence array.
        """
        if do_classify:
            return np.array(
                [
                    self.target_prob_generator.unred_state_to_id(s)
                    for s in gensample.state_seq
                ]
            )
        else:
            # temporary fix till we have a better solution
            # it could be then we need to have the target also generated by the generator
            if gensample.class_seq[-1] == 0:
                cls_seq = gensample.class_seq
                cls_seq[np.where(cls_seq == 0)[0][:-1] + 1] = 0
                return np.append(cls_seq[1:], [0])
            else:
                cls_seq = gensample.class_seq
                cls_seq[np.where(cls_seq == 0)[0] + 1] = 0
                return np.append(cls_seq[1:], [0])
            # return np.append(gensample.class_seq[1:], [0])


class PadSequence:
    """
    Collate function for padding sequences in batches.

    This class provides padding functionality for sequence data, supporting both
    classification and prediction tasks. It pads sequences to the same length within
    a batch and returns dictionaries with padded tensors and metadata.

    Attributes:
        pad_index: Index value used for padding sequences (default: -1).
        debug_class: Whether to include debug class sequences in output (default: True).
        pad_collate_fn: The collate function to use (set based on do_classify).

    Example:
        >>> pad_fn = PadSequence(do_classify=True, pad_index=-1)
        >>> batch = pad_fn([sample1, sample2, sample3])
        >>> # Returns dict with 'data', 'labels', 'mask', 'lens', 'gap_mask', etc.
    """

    def __init__(self, do_classify=None, pad_index=-1, debug_class=True):
        """
        Initialize the PadSequence collate function.

        Args:
            do_classify: If True, use classification collate function (includes labels).
                        If False, use prediction collate function (includes target_probs).
                        If None, must be set later (default: None).
            pad_index: Index value to use for padding sequences (default: -1).
            debug_class: Whether to include debug class sequences in output (default: True).
        """
        self.pad_index = pad_index
        self.debug_class = debug_class
        if do_classify:
            self.pad_collate_fn = self.pad_collate_classify
        else:
            self.pad_collate_fn = self.pad_collate_predict

    def __call__(self, batch):
        """
        Collate a batch of samples by padding sequences.

        Args:
            batch: List of samples from SeqDataset.__getitem__.

        Returns:
            dict: Dictionary containing padded tensors and metadata.
        """
        return self.pad_collate_fn(batch)

    def pad_collate_classify(self, batch):
        """
        Collate function for classification tasks.

        Pads sequences in a batch for classification, including data, labels, masks,
        gap masks, and optionally debug class sequences.

        Args:
            batch: List of tuples (data, target, class_seq, gap_mask) from SeqDataset.

        Returns:
            dict: Dictionary containing:
                - 'data': Padded data tensor (batch_size, max_len, features).
                - 'labels': Padded label tensor (batch_size, max_len).
                - 'mask': Padding mask tensor (batch_size, max_len).
                - 'lens': Sequence lengths tensor (batch_size,).
                - 'gap_mask': Padded gap mask tensor (batch_size, max_elem, max_len).
                - 'debug_class_seq': Padded debug class sequence (if debug_class=True).
        """
        data = []
        target = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(torch.Tensor(gap_mask_b))
            debug_class_seq.append(class_seq_b)

        max_len = max(tensor.size(1) for tensor in gap_masks)
        max_elem = max(tensor.size(0) for tensor in gap_masks)
        padded_gaps = [
            F.pad(tensor, (0, max_len - tensor.size(1), 0, max_elem - tensor.size(0)))
            for tensor in gap_masks
        ]
        gap_mask = torch.stack(padded_gaps)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(
            target, batch_first=True, padding_value=self.pad_index
        )
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)

        if self.debug_class:
            debug_class_seq = torch.Tensor(np.array(debug_class_seq))
            debug_class_seq = torch.nn.utils.rnn.pad_sequence(
                debug_class_seq, batch_first=True
            )

            return {
                "data": data.float(),
                "labels": target.long(),
                "mask": mask.float(),
                "lens": lens,
                "gap_mask": gap_mask.contiguous(),
                "debug_class_seq": debug_class_seq,
            }
        else:
            return {
                "data": data.float(),
                "labels": target.long(),
                "mask": mask.float(),
                "lens": lens,
                "gap_mask": gap_mask.contiguous(),
            }

    def pad_collate_predict(self, batch):
        """
        Collate function for prediction tasks.

        Pads sequences in a batch for prediction, including data, labels, target
        probabilities, masks, gap masks, and debug class sequences.

        Args:
            batch: List of tuples (data, target, class_seq, target_probs, gap_mask)
                from SeqDataset.

        Returns:
            dict: Dictionary containing:
                - 'data': Padded data tensor (batch_size, max_len, features).
                - 'labels': Padded label tensor (batch_size, max_len).
                - 'target_probs': Padded target probabilities (batch_size, max_len, num_classes).
                - 'mask': Padding mask tensor (batch_size, max_len).
                - 'lens': Sequence lengths tensor (batch_size,).
                - 'gap_mask': Padded gap mask tensor (batch_size, max_len).
                - 'debug_class_seq': Padded debug class sequence.
        """
        data = []
        target = []
        target_probs = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, target_probs_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            target_probs.append(target_probs_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(gap_mask_b)
            debug_class_seq.append(class_seq_b)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(
            target, batch_first=True, padding_value=self.pad_index
        )
        target_probs = torch.nn.utils.rnn.pad_sequence(target_probs, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)
        gap_masks = torch.nn.utils.rnn.pad_sequence(gap_masks, batch_first=True)
        debug_class_seq = torch.Tensor(np.array(debug_class_seq))
        debug_class_seq = torch.nn.utils.rnn.pad_sequence(
            debug_class_seq, batch_first=True
        )

        return {
            "data": data.float(),
            "labels": target.long(),
            "target_probs": target_probs.float(),
            "mask": mask.float(),
            "lens": lens,
            "gap_mask": gap_masks,
            "debug_class_seq": debug_class_seq,
        }
