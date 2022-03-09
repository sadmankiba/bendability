from __future__ import annotations
from typing import Iterable, Any

import regex as re
import numpy as np
from nptyping import NDArray

from util.util import rev_comp
from util.custom_types import DNASeq, KMerSeq


class KMer:
    @classmethod
    def find_pos(cls, seq: DNASeq, kmers: Iterable[str]) -> dict[str, list[int]]:
        return dict(
            map(
                lambda kmer: (
                    kmer,
                    [m.start() for m in re.finditer(kmer, seq, overlapped=True)],
                ),
                kmers,
            )
        )

    @classmethod
    def count_w_rc(cls, kmer: str, seqs: list[str]) -> NDArray[(Any,), int]:
        return np.array(cls.count(kmer, seqs)) + np.array(
            cls.count(kmer, rev_comp(seqs))
        )

    @classmethod
    def count(cls, kmer: str, seq: str | Iterable[str]) -> int | list[int]:
        if isinstance(seq, str):
            return len(re.findall(kmer, seq, overlapped=True))
        elif isinstance(seq, Iterable):
            return list(map(lambda c: cls.count(kmer, c), seq))

