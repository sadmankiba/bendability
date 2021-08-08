from __future__ import annotations
from custom_types import YeastChrNum

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt

import math
import random
import string
import re as bre  # built-in re
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
    rdf['Sequence'] = df['Sequence'].apply(
        lambda seq: reverse_compliment_of(seq))
    return pd.concat([df, rdf], ignore_index=True)


def sorted_split(df: pd.DataFrame,
                 n=1000,
                 n_bins=1,
                 ascending=False) -> list[pd.DataFrame]:
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

    return [
        sorted_df.iloc[start_pos:start_pos + math.ceil(n / n_bins)]
        for start_pos in range(0, n, math.ceil(n / n_bins))
    ]


def cut_sequence(df, start, stop):
    """
    Cut a sequence from start to stop position.
    start, stop - 1-indexed, inclusive
    """
    df['Sequence'] = df['Sequence'].str[start - 1:stop]
    return df


def get_possible_seq(size):
    """
    Generates all possible nucleotide sequences of particular length

    Returns:
        A list of sequences
    """

    possib_seq = ['']

    for _ in range(size):
        possib_seq = [
            seq + nc for seq in possib_seq for nc in ['A', 'C', 'G', 'T']
        ]

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
    alphabet = [chr(ord('a') + i) for i in range(n_letters)]

    for _ in range(size):
        possib_seq = [seq + c for seq in possib_seq for c in alphabet]

    return possib_seq


def find_shape_occurence_individual(df: pd.DataFrame, k_list: list,
                                    n_letters: int):
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
        df = df.assign(new_column=lambda x: x['Sequence'].str.count(seq))
        df = df.rename(columns={'new_column': seq})

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


def roman_to_num(chr_num: YeastChrNum) -> int:
    rom_num_map = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VIII': 8,
        'IX': 9,
        'X': 10,
        'XI': 11,
        'XII': 12,
        'XIII': 13,
        'XIV': 14,
        'XV': 15,
        'XVI': 16
    }
    return rom_num_map[chr_num]


class IOUtil:
    # TODO: Change name - SaveUtil
    def save_figure(self, path_str: str | Path):
        path = Path(path_str)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().set_size_inches(12, 6)
        plt.savefig(path, dpi=200)

    def save_tsv(self, df: pd.DataFrame, path_str: str | Path) -> None:
        """Save a dataframe in tsv format"""
        self.make_parent_dirs(path_str)
        df.to_csv(path_str, sep='\t', index=False, float_format='%.3f')

    def make_parent_dirs(self, path_str: str | Path) -> None:
        path = Path(path_str)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)

    def append_tsv(self, df: pd.DataFrame, path_str: str | Path) -> None:
        """Append a dataframe to a tsv if it exists, otherwise create"""
        path = Path(path_str)
        if path.is_file():
            target_df = pd.read_csv(path, sep='\t')
            pd.concat([df, target_df], join='outer', ignore_index=True)\
                .to_csv(path, sep='\t', index=False, float_format='%3f')
            return

        self.save_tsv(df, path_str)
