# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence generator utilities for creating synthetic sequence data.
"""

import logging
from dataclasses import dataclass

import numpy as np

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
        params,
        sequencer,
        compute_te=False,
        plot_transition_table=False
    ):
        self.params = params
        # Access data_generation parameters with nested structure
        data_gen = params['data_generation']
        self.seq_len = data_gen['seq_len_max']
        self.seed = params['seed']
        self.compute_te = compute_te
        self.plot_transition_table = plot_transition_table
        self.n_max_tries = 1e4  # number of maximum attempts to generate a string of correct length
        self.num_illustration_seq = 4
        
        self.combine_sequences = data_gen['combine_sequences']
        self.combined_seq_len = data_gen['combined_seq_length']

        logger.info(f"Task RNG seed {self.seed}")

        self.sequencer = sequencer

        # compute TE, correct needs to be set to True
        if self.compute_te:
            transition_table = (self.sequencer.transition_table(correct=True,
                                                                display=False) > 0).astype(int)
            TE = self.sequencer.topological_entropy(transitions=transition_table,
                                                    method='direct')
            try:
                import wandb
                wandb.log({'TE': TE})
            except:
                pass

            print(f"TE:\t{TE}")
            print("####")

        # plot transition table
        if self.plot_transition_table:
            import matplotlib.pyplot as plt
            from seqbench.seq_utils.markov_chain import MarkovChain

            P = self.sequencer.transition_table(correct=False,
                                                display=True).T
            mc = MarkovChain(P, self.sequencer.states,
                             node_fontsize=10,
                             node_radius=1.,
                             fontsize=10)
            #fig = mc.draw(title=f"start states: {grammar['start_states']}, TE: {round(TE, 2)}")

            for _ in range(self.num_illustration_seq):
                x = self.generate_sequence()
                seqs = x.state_seq
                print(seqs)

            start_states = [str(item) for item in self.params['gramm']['start_states']]
            try:
                mc.draw(title=f"start states: {start_states}, TE: {round(TE, 2)}",
                        figsize=(5,5))
            except:
                mc.draw(title=f"start states: {start_states}",
                        figsize=(5,5))

            fname = "grammar" 
            path = "."
            #TE_r = round(TE, 2)
            print(f'Save {path}/{fname}.pdf and {path}/{fname}.png')
            plt.savefig(f'{path}/{fname}.pdf')
            plt.savefig(f'{path}/{fname}.png', dpi=300)

            plt.close()
 
    def generate(self, idx, compute_length=True):
        
        if self.seed is not None:
            self.sequencer.rng = np.random.default_rng(self.seed * idx)
        else:
            self.sequencer.rng = np.random.default_rng()

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

        sequence = []
        cnt = 0
        data_gen = self.params['data_generation']
        while not (data_gen['seq_len_min'] <= len(sequence) <= data_gen['seq_len_max']):
            sequence = self.sequencer.generate_string(max_length=data_gen['seq_len_max'], as_states=True)

            #min_length: int = 0,
            #max_length: int = int(1e4),
            #length_range: tuple | None = None,
            #remove_eos: bool = True,
            #as_states: bool = False,
            #max_iter: int = int(1e3),

            #assert isinstance(sequence, tuple)
            
            sequence #= sequence[0] 

            if cnt > self.n_max_tries:
                raise RuntimeError("Could not generate a string of wanted length!")
            cnt += 1

        if not self.combine_sequences:
            assert self.__is_valid(sequence)

        state_seq = sequence + ['#']
        state_seq = [sym.replace("(", "").replace(")", "") for sym in state_seq]

        #sequence = sequence[:-1] # If A0 B1 C2 # -> A0 B1 C2
        #TODO: code doesn't support arbitrary class indices, it supports only A, B, C, a, b, c, ...
        class_indices = [sym[0] if len(sym) > 1 else sym for sym in  sequence] # A0 B1 C2 -> A B C
        class_indices = [ord(s)-64 if s.isupper() else ord(s)-70 for s in class_indices] + [0] # A B C -> 1 2 3 0

        assert len(class_indices) == len(state_seq)

        class_indices = np.array(class_indices)
        state_seq = np.array(state_seq)

        return GeneratorSample(
            class_indices,
            state_seq,
            0
        )

    def __is_valid(self, sequence):
        if sequence[-1] != '#':
            return False
        return True 
