from __future__ import annotations
from typing import TypedDict, Union

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from .dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode
from util.util import rev_comp
from util.custom_types import DNASeq


class SeqTarget(TypedDict):
    all_seqs: list[DNASeq]
    rc_seqs: list[DNASeq]
    target: Union[np.ndarray, None]


class OheResult(TypedDict):
    forward: np.ndarray
    reverse: np.ndarray
    target: Union[np.ndarray, None]


class Preprocess:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _get_sequences_target(self) -> SeqTarget:
        all_seqs = self._df["Sequence"].tolist()
        rc_seqs = [rev_comp(seq) for seq in all_seqs]
        target = self._df["C0"].to_numpy() if "C0" in self._df else None

        return {"all_seqs": all_seqs, "target": target, "rc_seqs": rc_seqs}

    def one_hot_encode(self) -> OheResult:
        def _seq_to_col_mat(seq):
            return np.array(list(seq)).reshape(-1, 1)

        one_hot_encoder = OneHotEncoder(categories="auto")
        one_hot_encoder.fit(_seq_to_col_mat("ACGT"))

        seq_and_target = self._get_sequences_target()

        forward = [
            one_hot_encoder.transform(_seq_to_col_mat(s)).toarray()
            for s in seq_and_target["all_seqs"]
        ]
        reverse = [
            one_hot_encoder.transform(_seq_to_col_mat(s)).toarray()
            for s in seq_and_target["rc_seqs"]
        ]

        return {
            "forward": np.stack(forward),
            "reverse": np.stack(reverse),
            "target": seq_and_target["target"],
        }

    def dinucleotide_encode(self):
        # Requires fix
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        forward_sequences = mono_to_dinucleotide(new_fasta)
        reverse_sequences = mono_to_dinucleotide(rc_fasta)

        forward = dinucleotide_one_hot_encode(forward_sequences)
        reverse = dinucleotide_one_hot_encode(reverse_sequences)

        dict = {}
        dict["readout"] = self.read_readout()
        dict["forward"] = forward
        dict["reverse"] = reverse
        return dict
