from __future__ import annotations

import pandas as pd
import numpy as np
import regex as re
import swifter

import math
import random
import string
import re as bre    # built-in re
import itertools as it
from pathlib import Path

def reverse_compliment_of(seq: str):
    # Define replacements
    rep = {"A": "T", "T": "A", 'G': 'C', 'C': 'G'} 
    
    # Create regex pattern
    rep = dict((bre.escape(k), v) for k, v in rep.items())
    pattern = bre.compile("|".join(rep.keys()))
    
    # Replace and return reverse sequence 
    return (pattern.sub(lambda m: rep[bre.escape(m.group(0))], seq))[::-1]


def append_reverse_compliment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends reverse compliment sequences to a dataframe
    """
    rdf = df.copy()
    rdf['Sequence'] = df['Sequence'].apply(lambda seq: reverse_compliment_of(seq))
    return pd.concat([df, rdf], ignore_index=True)


def sorted_split(df: pd.DataFrame, n=1000, n_bins=1, ascending=False) -> list[pd.DataFrame]:
    """
    Sort data according to C0 value
    params:
        df: dataframe
        n: Top n data to use
        n_bins: Number of bins(dataframes) to split to
        ascending: sort order
    returns:
        A list of dataframes.
    """
    sorted_df = df.sort_values(by=['C0'], ascending=ascending)[0:n]

    return [ sorted_df.iloc[start_pos : start_pos + math.ceil(n / n_bins)]
                for start_pos in range(0, n, math.ceil(n / n_bins)) ]


def cut_sequence(df, start, stop):
    """
    Cut a sequence from start to stop position.
    start, stop - 1-indexed, inclusive
    """
    df['Sequence'] = df['Sequence'].str[start-1:stop]
    return df


def get_possible_seq(size):
    """
    Generates all possible nucleotide sequences of particular length

    Returns:
        A list of sequences
    """

    possib_seq = ['']

    for _ in range(size):
        possib_seq = [ seq + nc for seq in possib_seq for nc in ['A', 'C', 'G', 'T'] ]

    return possib_seq


def get_possible_shape_seq(size: int, n_letters: int):
    """
    Generates all possible strings of particular length from an alphabet

    Args:
        size: Size of string
        n_letters: Number of letters to use

    Returns:
        A list of strings
    """
    possib_seq = ['']
    alphabet = [ chr(ord('a') + i) for i in range(n_letters)]

    for _ in range(size):
        possib_seq = [ seq + c for seq in possib_seq for c in alphabet ]

    return possib_seq


class HelicalSeparationCounter:
    """
    Functions for counting helical separation
    """
    def __init__(self):
        self._expected_dist_file = 'data/generated_data/helical_separation/expected_dist.tsv'

    def _get_all_dist(self, seq: str, nc_pair: tuple[str, str]) -> np.ndarray:
        """
        Returns 1D numpy array of size 48
        """ 
        pos_one = [ m.start() for m in re.finditer(nc_pair[0], seq, overlapped=True)]
        pos_two = [ m.start() for m in re.finditer(nc_pair[1], seq, overlapped=True)]
        pair_dist = [ abs(pos_pair[0] - pos_pair[1]) for pos_pair in it.product(pos_one, pos_two) ]
        return np.bincount(pair_dist, minlength=49)[1:] 


    def _count_helical_separation(self, seq: str, nc_pair: tuple[str, str]) -> int:
        """
        Count helical separation for an nc-pair in a single sequence
        """
        pair_all_dist = self._get_all_dist(seq, nc_pair)

        # helical
        at_helical_dist = 0
        for i in [10, 20, 30]:
            occur = pair_all_dist[i-2:i+1]
            # occur should be normalized
            at_helical_dist += occur.max()

        # half-helical
        at_half_helical_dist = 0
        for i in [5, 15, 25]:
            occur = pair_all_dist[i-2:i+1]
            at_half_helical_dist += occur.max()

        return at_helical_dist - at_half_helical_dist


    def _count_normalized_helical_separation(self, seq: str, nc_pair: tuple[str, str]) -> int:
        """
        Count normalized helical separation for an nc-pair in a single sequence
        """
        pair_all_dist = self._get_all_dist(seq, nc_pair)
        
        # Normalize dist
        expected_dist_df = pd.read_csv(self._expected_dist_file, sep='\t')
        pair_expected_dist = expected_dist_df.loc[
            expected_dist_df['Pair'] == f'{nc_pair[0]}-{nc_pair[1]}',
            expected_dist_df.columns[:-1]
        ].to_numpy().ravel()
        assert pair_expected_dist.shape == (48,), f'{pair_expected_dist.shape} is not (48,)'
        pair_normalized_dist = pair_all_dist / pair_expected_dist

        # helical
        at_helical_dist = 0
        for i in [10, 20, 30]:
            at_helical_dist += pair_normalized_dist[i-2:i+1].max()

        # half-helical
        at_half_helical_dist = 0
        for i in [5, 15, 25]:
            at_half_helical_dist += pair_normalized_dist[i-2:i+1].max()

        return at_helical_dist - at_half_helical_dist


    def count_dist_random_seq(self) -> pd.DataFrame:
        """"""
        p = Path(self._expected_dist_file)
        if p.is_file():
            return pd.read_csv(p, sep='\t')

        # Generate 10000 random sequences 
        sequences = gen_random_sequences(10000)

        # Generate dnc pairs 
        all_dinc = get_possible_seq(2)
        possib_pairs = [ pair for pair in it.combinations(all_dinc, 2)] + [(dinc, dinc) for dinc in all_dinc]
        assert len(possib_pairs) == 136

        # Count dist for 136 pairs in generated sequences
        mean_dist = np.array(list(map(lambda p: self._get_all_dist(p[0], p[1]), it.product(sequences, possib_pairs))))\
            .reshape(10000, 136, 48)\
            .mean(axis=0)
        assert mean_dist.shape == (136, 48)

        # Save a dataframe of 136 rows x 49 columns 
        df = pd.DataFrame(mean_dist, columns=np.arange(48)+1)
        df['Pair'] = list(map(lambda p: f'{p[0]}-{p[1]}', possib_pairs))
        df.to_csv(self._expected_dist_file, sep='\t', index=False)
        return df


    def find_helical_separation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find helical separation extent of all possible dinucleotide pairs for individual DNA sequences.

        Args:
            df: column `Sequence` contains DNA sequences

        Returns:
            A dataframe with columns added for all possible dinucleotide pairs.
        """
        df = df.copy()
        all_dinc = get_possible_seq(2)
        possib_pairs = [ pair for pair in it.combinations(all_dinc, 2)] + [(dinc, dinc) for dinc in all_dinc]
        assert len(possib_pairs) == 136

        for pair in possib_pairs:
            df[f'{pair[0]}-{pair[1]}'] = df['Sequence'].swifter.apply(lambda x: self._count_normalized_helical_separation(x, pair))

        return df


def find_shape_occurence_individual(df: pd.DataFrame , k_list: list, n_letters: int):
    """
    Find occurences of all possible shape sequences for individual DNA sequences.

    Args:
        df: column `Sequence` contains DNA shape sequences
        k_list: list of unit sizes to consider
        n_letters: number of letters used to encode shape

    Returns:
        A dataframe with columns added for all considered unit nucleotide sequences.
    """
    possib_seq = []
    for k in k_list:
        possib_seq += get_possible_shape_seq(k, n_letters)

    for seq in possib_seq:
        df = df.assign(new_column = lambda x: x['Sequence'].str.count(seq))
        df = df.rename(columns = {'new_column': seq})

    return df


def gen_random_sequences(n: int):
    """Generates n 50 bp random DNA sequences"""
    seq_list = []
    for _ in range(n):
        seq = ''

        for _ in range(50):
            d = random.random()
            if d < 0.25:
                c = 'A'
            elif d < 0.5:
                c = 'T'
            elif d < 0.75:
                c = 'G'
            else:
                c = 'C'

            seq += c

        seq_list.append(seq)

    return seq_list


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
