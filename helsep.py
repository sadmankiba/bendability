from __future__ import annotations

from util import get_possible_seq

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

