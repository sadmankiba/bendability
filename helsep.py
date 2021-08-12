from __future__ import annotations

from util import IOUtil, get_possible_seq, gen_random_sequences

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

# Constants
NUM_DINC_PAIRS = 136  # Possible dinucleotide pairs
NUM_DISTANCES = 48  # Possible distances between two dinucleotides


class HelicalSeparationCounter:
    """
    Helper class for counting helical separation
    """
    def __init__(self):
        self._expected_dist_file = 'data/generated_data/helical_separation/expected_p.tsv'
        self._dinc_pairs = self._get_dinc_pairs()

    def _get_dinc_pairs(self) -> list[tuple[str, str]]:
        """
        Generates dinucleotide pairs
        
        Returns:
            A list of tuples of two dinucleotides
        """
        all_dinc = get_possible_seq(2)

        dinc_pairs = [pair for pair in it.combinations(all_dinc, 2)
                      ] + [(dinc, dinc) for dinc in all_dinc]
        assert len(dinc_pairs) == 136

        return dinc_pairs

    def _get_all_dist(self, seq: str) -> np.ndarray:
        """
        Find unnormalized p(i) for i = 1-48 for all dinucleotide pairs

        Returns:
            A 2D numpy array of shape (136, 48)
        """
        all_dinc = get_possible_seq(2)

        # Find positions of all dinucleotides in sequence
        pos_dinc = dict(
            map(
                lambda dinc:
                (dinc,
                 [m.start() for m in re.finditer(dinc, seq, overlapped=True)]),
                all_dinc))

        def find_pair_dist(pos_one: list[int],
                           pos_two: list[int]) -> list[int]:
            """
            Find absolute distances from positions

            Example - Passing parameters [3, 5] and [1, 2] will return [2, 1, 4, 3] 
            """
            return [
                abs(pos_pair[0] - pos_pair[1])
                for pos_pair in it.product(pos_one, pos_two)
            ]

        # Calculate absolute distances for all dinucleotide pairs
        dinc_dists: list[np.ndarray] = map(
            lambda p: find_pair_dist(pos_dinc[p[0]], pos_dinc[p[1]]),
            self._dinc_pairs)

        # Calculate unnormalized p(i) for i=1-48 from abs distances
        result = np.array(
            list(
                map(
                    lambda one_pair_dist: np.bincount(one_pair_dist,
                                                      minlength=49)[1:],
                    dinc_dists)))
        return result

    def _get_all_normalized_dist(self, seq_list: list[str]) -> np.ndarray:
        """
        Calculates normalized p(i) for i = 1-48 for all dinc pairs

        Returns:
            A numpy array of shape (len(seq_list), 136, 48)
        """
        # Find all pair distance in sequence list
        pair_all_dist = np.array(list(map(self._get_all_dist, seq_list)))
        assert pair_all_dist.shape == (len(seq_list), 136, 48)

        # Load expected distance
        expected_dist_df = self.calculate_expected_p()
        expected_dist = expected_dist_df.drop(columns='Pair').values
        assert expected_dist.shape == (136, 48)

        return pair_all_dist / expected_dist

    def _normalized_helical_sep_of(self, seq_list: list[str]) -> np.ndarray:
        """
        Count normalized helical separation for an nc-pair in a single sequence

        Vectorized calculation!

        Returns:
            A numpy array of shape (len(seq_list), 136)
        """
        pair_normalized_dist = self._get_all_normalized_dist(seq_list)

        def sum_max_p(pos_list: list[int]):
            """
            Adds max p(i) around given positions

            Args:
                list of positions 
            """
            pass
            hel_arr = np.array(
                list(
                    map(
                        lambda i: np.max(
                            pair_normalized_dist[:, :, i - 2:i + 1], axis=2),
                        pos_list)))
            assert hel_arr.shape == (len(pos_list), len(seq_list), 136)
            return np.sum(hel_arr, axis=0)

        at_helical_dist = sum_max_p([10, 20, 30])
        at_half_helical_dist = sum_max_p([5, 15, 25])

        return at_helical_dist - at_half_helical_dist

    def calculate_expected_p(self) -> pd.DataFrame:
        """
        Calculates expected p(i) of dinucleotide pairs.
        """
        if Path(self._expected_dist_file).is_file():
            return pd.read_csv(self._expected_dist_file, sep='\t')

        # Generate 10000 random sequences
        seq_list = gen_random_sequences(10000)

        # Count mean distance for 136 pairs in generated sequences
        pair_all_dist = np.array(list(map(self._get_all_dist, seq_list)))
        mean_pair_dist = pair_all_dist.mean(axis=0)
        assert mean_pair_dist.shape == (136, 48)

        # Save a dataframe of 136 rows x 49 columns
        df = pd.DataFrame(mean_pair_dist, columns=np.arange(48) + 1)
        df['Pair'] = list(map(lambda p: f'{p[0]}-{p[1]}', self._dinc_pairs))
        IOUtil().save_tsv(df, self._expected_dist_file)
        return df

    def find_helical_separation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find helical separation extent of all possible dinucleotide pairs for DNA sequences.

        Args:
            df: column `Sequence` contains DNA sequences

        Returns:
            A dataframe with columns added for all possible dinucleotide pairs.
        """
        df = df.copy()
        df[self._dinc_pairs] = self._normalized_helical_sep_of(
            df['Sequence'].tolist())

        return df

    def plot_normalized_dist(self, df: pd.DataFrame,
                             library_name: str) -> None:
        """
        Plots avg. normalized distance of sequences with most and least 1000 C0 values 
        """
        most_1000 = df.sort_values('C0').iloc[:1000]
        least_1000 = df.sort_values('C0').iloc[-1000:]

        most1000_dist = self._get_all_normalized_dist(
            most_1000['Sequence'].tolist()).mean(axis=0)
        least1000_dist = self._get_all_normalized_dist(
            least_1000['Sequence'].tolist()).mean(axis=0)

        assert most1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)
        assert least1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)

        for i in range(NUM_DINC_PAIRS):
            plt.close()
            plt.clf()
            pair_str = self._dinc_pairs[i][0] + "-" + self._dinc_pairs[i][1]
            plt.plot(np.arange(NUM_DISTANCES) + 1,
                     most1000_dist[i],
                     linestyle='-',
                     color='r',
                     label='1000 most loopable sequences')
            plt.plot(np.arange(NUM_DISTANCES) + 1,
                     least1000_dist[i],
                     linestyle='-',
                     color='b',
                     label='1000 least loopable sequences')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20))
            plt.xlabel('Position (bp)')
            plt.ylabel('p(i)')
            plt.title(pair_str)
            plt.savefig(f'figures/distances/{library_name}/{pair_str}.png',
                        bbox_inches="tight")
