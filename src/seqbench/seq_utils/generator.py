# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence generator utilities for creating synthetic sequence data.

This module is the boundary between SeqBench and any external trial source
(notably symseq generators). A ``source`` here is duck-typed against the
``TrialSource`` Protocol: it must expose ``alphabet`` and ``draw_trial()``.

Token → class-index conversion is handled by :class:`SymbolEncoder` rather
than the old hardcoded ``ord(s) - 64`` arithmetic, so any alphabet works.
"""

import logging
from dataclasses import dataclass

import numpy as np

from seqbench.seq_utils.symbol_encoder import SymbolEncoder

logger = logging.getLogger('generator')


def determine_decimal_digits(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def rotate(l, n):
    return l[n:] + l[:n]

@dataclass
class GeneratorSample:
    class_seq:          list[int]
    state_seq:          list[str]
    length:    int

class SequenceGenerator:

    def __init__(
        self,
        source,
        *,
        seq_len_min: int,
        seq_len_max: int,
        combine_sequences: bool,
        combined_seq_len: int,
        seed=None,
        compute_te=False,
        plot_transition_table=False,
    ):
        self.seq_len_min = seq_len_min
        self.seq_len = seq_len_max
        self.seed = seed
        self.compute_te = compute_te
        self.plot_transition_table = plot_transition_table
        self.n_max_tries = 1e4  # number of maximum attempts to generate a string of correct length
        self.num_illustration_seq = 4

        self.combine_sequences = combine_sequences
        self.combined_seq_len = combined_seq_len

        logger.info(f"Task RNG seed {self.seed}")

        self.source = source
        self.encoder = SymbolEncoder(source.alphabet)

        # Topological entropy — only available for grammar sources.
        if self.compute_te:
            if hasattr(self.source, 'transition_table') and hasattr(self.source, 'topological_entropy'):
                transition_table = (self.source.transition_table(correct=True,
                                                                 display=False) > 0).astype(int)
                TE = self.source.topological_entropy(transitions=transition_table,
                                                     method='direct')
                try:
                    import wandb
                    wandb.log({'TE': TE})
                except:
                    pass

                print(f"TE:\t{TE}")
                print("####")
            else:
                logger.info("compute_te requested but source has no transition_table; skipping.")

        # Transition-table plot — only available for grammar sources.
        if self.plot_transition_table:
            if hasattr(self.source, 'transition_table') and hasattr(self.source, 'states'):
                import matplotlib.pyplot as plt
                from seqbench.seq_utils.markov_chain import MarkovChain

                P = self.source.transition_table(correct=False,
                                                 display=True).T
                mc = MarkovChain(P, self.source.states,
                                 node_fontsize=10,
                                 node_radius=1.,
                                 fontsize=10)

                for _ in range(self.num_illustration_seq):
                    x = self.generate_sequence()
                    seqs = x.state_seq
                    print(seqs)

                start_states = (
                    [str(item) for item in self.source.start_states]
                    if hasattr(self.source, 'start_states') else []
                )
                try:
                    mc.draw(title=f"start states: {start_states}, TE: {round(TE, 2)}",
                            figsize=(5,5))
                except:
                    mc.draw(title=f"start states: {start_states}",
                            figsize=(5,5))

                fname = "grammar"
                path = "."
                print(f'Save {path}/{fname}.pdf and {path}/{fname}.png')
                plt.savefig(f'{path}/{fname}.pdf')
                plt.savefig(f'{path}/{fname}.png', dpi=300)

                plt.close()
            else:
                logger.info("plot_transition_table requested but source has no transition_table; skipping.")

    @classmethod
    def from_config(cls, config, source, **kwargs):
        """Construct from a legacy Config object (backward compat for seq_dataset.py)."""
        data_gen = config['data_generation']
        return cls(
            source,
            seq_len_min=data_gen['seq_len_min'],
            seq_len_max=data_gen['seq_len_max'],
            combine_sequences=data_gen['combine_sequences'],
            combined_seq_len=data_gen['combined_seq_length'],
            seed=config['seed'],
            **kwargs,
        )

    def generate(self, idx, compute_length=True):

        # Per-sample seeding. symseq generators expose a writable .rng;
        # other sources may choose to ignore this.
        if self.seed is not None:
            self.source.rng = np.random.default_rng(self.seed * idx)
        else:
            self.source.rng = np.random.default_rng()

        if self.combine_sequences:
            gen_sample = self.generate_sequences()
        else:
            gen_sample =  self.generate_sequence()

        if compute_length:
            gen_sample.length = self.__compute_sequence_length(gen_sample.class_seq)

        return gen_sample

    def __compute_sequence_length(self, seq):
        return len(seq)

    def generate_sequences(self):

        comb_sample = GeneratorSample(
            np.array([]),
            np.array([]),
            0
        )

        seq_len = 0

        while seq_len < self.combined_seq_len:
            sample = self.generate_sequence()
            comb_sample = self.__concat_samples(comb_sample, sample)
            seq_len = comb_sample.class_seq.shape[0]

        comb_sample.class_seq = comb_sample.class_seq[:self.combined_seq_len]
        comb_sample.state_seq = comb_sample.state_seq[:self.combined_seq_len]

        return comb_sample

    def __concat_samples(self, comb_sample, sample):
        comb_sample.class_seq = np.concatenate((comb_sample.class_seq, sample.class_seq))
        comb_sample.state_seq = np.concatenate((comb_sample.state_seq, sample.state_seq))

        comb_sample.class_seq = comb_sample.class_seq.astype(int)

        return comb_sample

    def generate_sequence(self):

        seq_len_min = self.seq_len_min
        seq_len_max = self.seq_len

        symbols: list[str] = []
        states: list[str] = []
        cnt = 0
        while not (seq_len_min <= len(symbols) <= seq_len_max):
            trial = self.source.draw_trial()
            symbols = list(trial.symbols)
            states = list(trial.states) if trial.states is not None else list(symbols)

            if cnt > self.n_max_tries:
                raise RuntimeError("Could not generate a string of wanted length!")
            cnt += 1

        # Append EOS marker to both views; encoder maps '#' -> 0.
        state_seq = states + [SymbolEncoder.EOS_SYMBOL]
        state_seq = [sym.replace("(", "").replace(")", "") for sym in state_seq]

        class_indices = self.encoder.encode(symbols) + [SymbolEncoder.EOS_INDEX]

        assert len(class_indices) == len(state_seq)

        class_indices = np.array(class_indices)
        state_seq = np.array(state_seq)

        return GeneratorSample(
            class_indices,
            state_seq,
            0
        )
