# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Generator Module for SeqBench

This module provides utilities for generating synthetic sequence datasets
with configurable parameters and parallel processing capabilities.

Key Components:
- DatasetGenerator: Main class for generating sequence datasets
- Static utility methods for file I/O operations
"""

# Standard library imports
import os
import yaml
from dataclasses import dataclass
from multiprocessing import Pool, Manager
from typing import Optional, Any, Dict

# Third-party imports
import numpy as np
import tqdm

# Local imports
from seqbench.seq_utils.generator import GeneratorSample


@dataclass
class MCState:
    """
    Markov Chain state representation.

    Attributes:
        node: Current state node identifier (str)
        childs: Array of possible child states (str array)
        probs: Array of transition probabilities to child states (float array)
    """

    node: str
    childs: np.ndarray  # str
    probs: np.ndarray  # float


class RestrictedTargetProbGenerator:
    """
    Generator for target probabilities with state restrictions.

    This class manages the generation of target probabilities for sequence
    prediction tasks while maintaining state transition constraints. It supports
    both reduced and unreduced state spaces for efficient probability computation.

    Attributes:
        num_reduced_states: Number of states in reduced state space
        num_unreduced_states: Number of states in unreduced state space
        transitions: State transition matrix
        transition_probs: Transition probability matrix
        red_state_to_id_map: Mapping from reduced states to IDs
        unred_state_to_id_map: Mapping from unreduced states to IDs
        id_to_red_state_map: Mapping from IDs to reduced states
        id_to_unred_state_map: Mapping from IDs to unreduced states
        reduced_states: List of reduced state identifiers
        unreduced_states: List of unreduced state identifiers
    """

    def __init__(self):
        self.num_reduced_states = 0
        self.num_unreduced_states = 0
        self.transitions = None
        self.transition_probs = None
        self.red_state_to_id_map = {}
        self.unred_state_to_id_map = {}
        self.id_to_red_state_map = {}
        self.id_to_unred_state_map = {}
        self.reduced_states = []
        self.unreduced_states = []

    def read_transitions_from_file(self, transitions_filepath):
        print(f"Reading transitions from {transitions_filepath}!")
        with open(transitions_filepath, "r") as file:
            transitions = DatasetGenerator.read_transitions_from_file(file)
        self.__parse(transitions)

    def read_transitions_from_generator(self, generator):
        print(f"Loading transitions from generator!")
        if not hasattr(generator.source, "transitions"):
            raise AttributeError(
                "read_transitions_from_generator requires a grammar source "
                "with a `.transitions` attribute (e.g. ArtificialGrammar)."
            )
        transitions = {}
        for transition in generator.source.transitions:
            s_from = transition[0]
            s_to = transition[1]
            prob = transition[2]

            if s_from not in transitions:
                transitions[s_from] = MCState(s_from, np.array([s_to]), np.array([prob]))
            else:
                transitions[s_from].childs = np.append(transitions[s_from].childs, s_to)
                transitions[s_from].probs = np.append(transitions[s_from].probs, prob)
        self.__parse(transitions)

    def __parse(self, transitions):
        self.transitions = transitions
        self.unreduced_states = list(transitions.keys())
        self.num_unreduced_states = len(self.unreduced_states)

        self.__parse_states()
        self.__parse_transitions(transitions)

    def __parse_states(self):
        tmp_id_to_state_map = {}
        for s in self.unreduced_states:
            s_red = self.reduce_state(s)
            if s_red not in self.reduced_states:
                s_red_id = ord(s_red) - 64 if s_red.isupper() else ord(s_red) - 70
                self.reduced_states.append(s_red)
                self.red_state_to_id_map[s_red] = s_red_id
                tmp_id_to_state_map[s_red_id] = s_red

        self.num_reduced_states = len(self.reduced_states)

        tmp_reduced_states_count = [0 for _ in range(self.num_reduced_states)]
        for s in self.unreduced_states:
            s_red = self.reduce_state(s)
            s_red_id = self.red_state_to_id_map[s_red]
            tmp_reduced_states_count[s_red_id - 1] += 1

        id_count = 1
        for s_red_id in range(1, self.num_reduced_states + 1):
            s_red = tmp_id_to_state_map[s_red_id]
            s_red_count = tmp_reduced_states_count[s_red_id - 1]
            if s_red_count == 1:
                self.unred_state_to_id_map[f"{s_red}"] = id_count
                self.unred_state_to_id_map[f"{s_red}0"] = id_count  # just in case
                id_count += 1
            else:
                for i in range(s_red_count):
                    self.unred_state_to_id_map[f"{s_red}{i}"] = id_count
                    id_count += 1

        # Account for eos symbol
        self.red_state_to_id_map["#"] = 0
        self.unred_state_to_id_map["#"] = 0
        self.num_reduced_states += 1
        self.num_unreduced_states += 1
        self.reduced_states.append("#")
        self.unreduced_states.append("#")
        self.id_to_red_state_map = {v: k for k, v in self.red_state_to_id_map.items()}
        self.id_to_unred_state_map = {v: k for k, v in self.unred_state_to_id_map.items()}

    def __parse_transitions(self, transitions):
        transition_probs = {}

        for s in self.unreduced_states:
            transition_probs[s] = np.zeros(self.num_reduced_states)

        for state_name, mcstate in transitions.items():
            for i, child_s in enumerate(mcstate.childs):
                child_s = self.reduce_state(child_s)
                child_s = self.red_state_to_id_map[child_s]
                transition_probs[state_name][child_s] += mcstate.probs[i]

        self.transition_probs = transition_probs

    def __call__(self, state_seq):
        assert self.transition_probs is not None

        target_probs = []
        for state in state_seq:
            if state == "#":
                x = np.zeros(self.num_reduced_states)
                x[0] = 1
                target_probs.append(x)
            else:
                transition_prob = self.transition_probs[state]
                target_probs.append(transition_prob)
        return np.array(target_probs)

    def reduce_state(self, s):
        assert len(s) <= 4, f"Unreduced states should not have more than 2 characters. Got {s}!"
        return s[0]

    def print_transitions(self):
        for s_from, state in self.transitions.items():
            print(s_from, state)

    def red_state_to_id(self, red_state):
        return self.red_state_to_id_map[red_state]

    def unred_state_to_id(self, unred_state):
        return self.unred_state_to_id_map[unred_state]

    def id_to_red_state(self, id):
        return self.id_to_red_state_map[id]

    def id_to_unred_state(self, id):
        return self.id_to_unred_state_map[id]

    def get_reduced_states_sorted(self):
        return sorted(self.reduced_states, key=lambda red_state: self.red_state_to_id(red_state))

    def get_unreduced_states_sorted(self):
        return sorted(self.unreduced_states, key=lambda unred_state: self.unred_state_to_id(unred_state))


class DatasetGenerator:
    """
    Generator for creating synthetic sequence datasets.

    This class handles the generation of training and test datasets with
    configurable parameters, parallel processing, and file I/O operations.

    Attributes:
        seq_generator: Sequence generator instance
        dataset_size: Number of samples to generate
        output_dir: Directory to save generated datasets
        config_file_path: Path to configuration file
        generate_train: Whether to generate training data
        generate_test: Whether to generate test data
        append_hash_to_output_dir: Whether to append hash to output directory
    """

    def __init__(
        self,
        seq_generator: Any,
        dataset_size: int = 0,
        output_dir: Optional[str] = None,
        config_file_path: Optional[str] = None,
        generate_train: bool = False,
        generate_test: bool = False,
        append_hash_to_output_dir: bool = False,
    ) -> None:
        """
        Initialize the DatasetGenerator.

        Args:
            seq_generator: Sequence generator instance
            dataset_size: Number of samples to generate
            output_dir: Directory to save generated datasets
            config_file_path: Path to configuration file
            generate_train: Whether to generate training data
            generate_test: Whether to generate test data
            append_hash_to_output_dir: Whether to append hash to output directory
        """
        self.seq_generator = seq_generator
        self.dataset_size = dataset_size
        self.output_dir = output_dir
        self.config_file_path = config_file_path
        self.generate_train = generate_train
        self.generate_test = generate_test
        self.append_hash_to_output_dir = append_hash_to_output_dir

    @staticmethod
    def create_gensample_from_str(line: str) -> GeneratorSample:
        """
        Create a GeneratorSample from a string representation.

        Args:
            line: String in format "class_seq::state_seq::length"

        Returns:
            GeneratorSample: Parsed sample object

        Raises:
            AssertionError: If class_seq and state_seq have different lengths
        """
        line = line.split("::")

        class_seq = line[0].replace("[", "").replace("]", "")
        class_seq = class_seq.split(",")
        class_seq = [int(d.strip()) for d in class_seq]

        state_seq = line[1].replace("[", "").replace("]", "")
        state_seq = state_seq.replace("'", "")
        state_seq = state_seq.split(",")
        state_seq = [d.strip() for d in state_seq]

        assert len(class_seq) == len(state_seq), "Class and state sequences must have same length"

        length = int(line[2])

        return GeneratorSample(class_seq, state_seq, length)

    @staticmethod
    def read_transitions_from_file(file) -> Dict[str, MCState]:
        """
        Read transition data from a file.

        Args:
            file: File object to read from

        Returns:
            Dictionary mapping state names to MCState objects
        """
        transitions = {}
        for line in file:
            line = line.replace("\n", "").strip()
            line = line.replace("(", "").replace(")", "")
            line = line.replace("'", "")
            line = line.split(",")
            s_from = line[0].strip()
            s_to = line[1].strip()
            prob = float(line[2].strip())

            if s_from not in transitions:
                transitions[s_from] = MCState(s_from, np.array([s_to]), np.array([prob]))
            else:
                transitions[s_from].childs = np.append(transitions[s_from].childs, s_to)
                transitions[s_from].probs = np.append(transitions[s_from].probs, prob)
        return transitions

    @staticmethod
    def write_gensample_to_file(file, gensample: GeneratorSample) -> None:
        """
        Write a GeneratorSample to a file.

        Args:
            file: File object to write to
            gensample: GeneratorSample to write

        Raises:
            AssertionError: If class_seq and state_seq have different lengths
        """
        class_seq = gensample.class_seq.tolist()
        state_seq = gensample.state_seq.tolist()

        assert len(class_seq) == len(state_seq), "Class and state sequences must have same length"

        file.write(f"{class_seq}::{state_seq}::{gensample.length}\n")

    def generate(self) -> None:
        """
        Generate the dataset according to configuration.

        Creates output directory and generates train/test datasets
        as specified in the configuration.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Generating in {self.output_dir}!")
        if self.generate_train:
            # self.__generate_for_dataset_parallelize('train', self.dataset_size)
            self.__generate_for_dataset("train", self.dataset_size)

        if self.generate_test:
            # self.__generate_for_dataset_parallelize('test', self.dataset_size)
            self.__generate_for_dataset("test", self.dataset_size)

        self.__write_config_to_file()
        self.__write_transitions_to_file()

    def _generate_single(self, args) -> None:
        """
        Generate a single sample (used for parallel processing).

        Args:
            args: Tuple containing (idx, generator, output_file, lock)
        """
        idx, generator, output_file, lock = args
        gensample = generator.generate(idx)
        # Use lock to safely write to shared file
        with lock:
            with open(output_file, "a") as f:
                DatasetGenerator.write_gensample_to_file(f, gensample)

    def __generate_for_dataset(self, filename: str, dataset_size: int) -> None:
        """
        Generate dataset sequentially.

        Args:
            filename: Name of the output file
            dataset_size: Number of samples to generate
        """
        print(f"Generating {filename}!")
        file = open(os.path.join(self.output_dir, filename), "w")
        for i in tqdm.tqdm(range(dataset_size)):
            gensample = self.seq_generator.generate(i)
            DatasetGenerator.write_gensample_to_file(file, gensample)
        file.close()

    def __generate_for_dataset_parallelize(self, filename: str, dataset_size: int) -> None:
        """
        Generate dataset using parallel processing.

        Args:
            filename: Name of the output file
            dataset_size: Number of samples to generate
        """
        print(f"Generating {filename}!")

        output_file = os.path.join(self.output_dir, filename)

        # Create a manager to share the lock between processes
        with Manager() as manager:
            lock = manager.Lock()

            # Clear the file before starting
            with open(output_file, "w") as f:
                pass

            # Create argument tuples for each task
            args_list = [(i, self.seq_generator, output_file, lock) for i in range(dataset_size)]

            # Use multiprocessing to generate and write samples in parallel
            with Pool() as pool:
                list(
                    tqdm.tqdm(
                        pool.imap(self._generate_single, args_list), total=dataset_size, desc="Generating samples"
                    )
                )

    def __write_config_to_file(self) -> None:
        """Write configuration to YAML file."""
        print(f"Writing config!")

        if self.config_file_path is None:
            config = {}
        else:
            with open(self.config_file_path, "r") as f:
                config = yaml.safe_load(f)

        with open(os.path.join(self.output_dir, "config.yaml"), "w") as file:
            file.write(yaml.safe_dump(config))

    def __write_transitions_to_file(self) -> None:
        """Write transition data to file."""
        print(f"Writing transitions!")
        with open(os.path.join(self.output_dir, "transitions"), "w") as file:
            for transition in self.seq_generator.source.transitions:
                file.write(f"{transition[0]}, {transition[1]}, {float(transition[2])}\n")
