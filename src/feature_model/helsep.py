from __future__ import annotations
import itertools as it
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from nptyping import NDArray

from chromosome.dinc import Dinc
from util.util import FileSave, PathObtain, get_possible_seq, gen_random_sequences
from util.custom_types import DNASeq, DiNc

# Constants
NUM_DINC_PAIRS = 136  # Possible dinucleotide pairs
NUM_DISTANCES = 48  # Possible distances between two dinucleotides
SEQ_COL = "Sequence"

class DincUtil:
    #TODO: Merge DincUtil with Dinc?
    @classmethod
    def pairs_all(cls) -> list[tuple[DiNc, DiNc]]:
        """
        Generates all possible dinucleotide pairs
        """
        all_dinc = get_possible_seq(2)

        dinc_pairs = [pair for pair in it.combinations(all_dinc, 2)] + [
            (dinc, dinc) for dinc in all_dinc
        ]
        assert len(dinc_pairs) == 136

        return dinc_pairs

    @classmethod
    def pair_str(cls, dinc_pair: tuple[DiNc, DiNc]) -> str:
        return dinc_pair[0] + "-" + dinc_pair[1]

class HelicalSeparationCounter:
    """
    Helper class for counting helical separation

    The hel sep of a given NN-NN pair in a given sequence = Sum_{i = 10, 20, 30}
    max(p(i-1), p(i), p(i+1)) - Sum_{i = 5, 15, 25} max(p(i-1), p(i), p(i+1))

    where p(i) is the pairwise distance distribution function, the number of
    times that the two dinucleotides in the pair are separated by a distance i
    in the sequence, normalized by an expected p(i) for the NN-NN in random
    seqs.
    """

    def __init__(self):
        self._expected_dist_file = (
            f"{PathObtain.data_dir()}/generated_data/helical_separation/expected_p.tsv"
        )
        self._dinc_pairs = DincUtil.pairs_all()

    def helical_sep_of(self, seq_list: list[DNASeq]) -> pd.DataFrame:
        """
        Count normalized helical separation
        """
        pair_dist_occur = self._dinc_pair_dist_occur_normd(seq_list)

        def _dist_occur_max_at(pos: int) -> NDArray[(Any, 136), float]:
            return np.max(pair_dist_occur[:, :, pos - 2 : pos + 1], axis=2)

        at_hel_dist = sum(list(map(_dist_occur_max_at, [10, 20, 30])))
        at_half_hel_dist = sum(list(map(_dist_occur_max_at, [5, 15, 25])))

        dinc_df = pd.DataFrame(
            at_hel_dist - at_half_hel_dist,
            columns=list(map(DincUtil.pair_str, self._dinc_pairs)),
        )
        dinc_df[SEQ_COL] = seq_list
        return dinc_df

    def _dinc_pair_dist_occur_normd(
        self, seq_list: list[DNASeq]
    ) -> NDArray[(Any, 136, 48), float]:
        """
        Calculates normalized p(i) for i = 1-48 for all dinc pairs

        """
        pair_dist_occur = np.array(list(map(self._pair_dinc_dist_in, seq_list)))
        assert pair_dist_occur.shape == (len(seq_list), 136, 48)

        exp_dist_occur = self.calculate_expected_p().drop(columns="Pair").values
        assert exp_dist_occur.shape == (136, 48)

        return pair_dist_occur / exp_dist_occur

    def calculate_expected_p(self) -> pd.DataFrame:
        """
        Calculates expected p(i) of dinucleotide pairs.
        """
        if Path(self._expected_dist_file).is_file():
            return pd.read_csv(self._expected_dist_file, sep="\t")

        # Generate 10000 random sequences
        seq_list = gen_random_sequences(10000)

        # Count mean distance for 136 pairs in generated sequences
        pair_all_dist = np.array(list(map(self._pair_dinc_dist_in, seq_list)))
        mean_pair_dist = pair_all_dist.mean(axis=0)
        assert mean_pair_dist.shape == (136, 48)

        # Save a dataframe of 136 rows x 49 columns
        df = pd.DataFrame(mean_pair_dist, columns=np.arange(48) + 1)
        df["Pair"] = list(map(lambda p: f"{p[0]}-{p[1]}", self._dinc_pairs))
        FileSave.tsv(df, self._expected_dist_file)
        return df

    def _pair_dinc_dist_in(
        self, seq: DNASeq
    ) -> NDArray[(NUM_DINC_PAIRS, NUM_DISTANCES), int]:
        """
        Find unnormalized p(i) for i = 1-48 for all dinucleotide pairs
        """
        pos_dinc = Dinc.find_pos(seq, get_possible_seq(2))

        dinc_dists: list[np.ndarray] = map(
            lambda p: self._find_pair_dist(pos_dinc[p[0]], pos_dinc[p[1]]),
            self._dinc_pairs,
        )

        return np.array(
            list(
                map(
                    lambda one_pair_dist: np.bincount(
                        one_pair_dist, minlength=NUM_DISTANCES + 1
                    )[1:],
                    dinc_dists,
                )
            )
        )

    @classmethod
    def _find_pair_dist(cls, pos_one: list[int], pos_two: list[int]) -> list[int]:
        """
        Find absolute distances from positions

        Example - Passing parameters [3, 5] and [1, 2] will return [2, 1, 4, 3]
        """
        return [
            abs(pos_pair[0] - pos_pair[1]) for pos_pair in it.product(pos_one, pos_two)
        ]

    def plot_normalized_dist(self, df: pd.DataFrame, library_name: str) -> None:
        """
        Plots avg. normalized distance of sequences with most and least 1000 C0 values
        """
        most_1000 = df.sort_values("C0").iloc[:1000]
        least_1000 = df.sort_values("C0").iloc[-1000:]

        most1000_dist = self._dinc_pair_dist_occur_normd(
            most_1000["Sequence"].tolist()
        ).mean(axis=0)
        least1000_dist = self._dinc_pair_dist_occur_normd(
            least_1000["Sequence"].tolist()
        ).mean(axis=0)

        assert most1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)
        assert least1000_dist.shape == (NUM_DINC_PAIRS, NUM_DISTANCES)

        for i in range(NUM_DINC_PAIRS):
            plt.close()
            plt.clf()
            pair_str = self._dinc_pairs[i][0] + "-" + self._dinc_pairs[i][1]
            plt.plot(
                np.arange(NUM_DISTANCES) + 1,
                most1000_dist[i],
                linestyle="-",
                color="r",
                label="1000 most loopable sequences",
            )
            plt.plot(
                np.arange(NUM_DISTANCES) + 1,
                least1000_dist[i],
                linestyle="-",
                color="b",
                label="1000 least loopable sequences",
            )
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20))
            plt.xlabel("Position (bp)")
            plt.ylabel("p(i)")
            plt.title(pair_str)
            plt.savefig(
                f"{PathObtain.figure_dir()}/distances/{library_name}/{pair_str}.png",
                bbox_inches="tight",
            )
