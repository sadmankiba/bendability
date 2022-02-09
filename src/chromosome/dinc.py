from __future__ import annotations
from typing import Iterable, Any

import regex as re
import numpy as np
import pandas as pd
from nptyping import NDArray

from chromosome.chromosome import Chromosome
from util.util import reverse_compliment_of
from util.custom_types import PosOneIdx, DNASeq, DiNc, KMerSeq


class KMerParent:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    @classmethod
    def _count_substr(cls, substr: str, cntnr: str) -> int:
        return len(re.findall(substr, cntnr, overlapped=True))


class Dinc(KMerParent):
    def __init__(self, chrm: Chromosome) -> None:
        super().__init__(chrm)

    @classmethod
    def find_pos(cls, seq: DNASeq, dincs: Iterable[DiNc]) -> dict[DiNc, list[int]]:
        return dict(
            map(
                lambda dinc: (
                    dinc,
                    [m.start() for m in re.finditer(dinc, seq, overlapped=True)],
                ),
                dincs,
            )
        )

    def ta_count_multisegment(
        self, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> list[int]:
        return list(map(lambda se: self.ta_count_singlesegment(*se), zip(starts, ends)))

    def cg_count_multisegment(
        self, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> list[int]:
        return list(map(lambda se: self.cg_count_singlesegment(*se), zip(starts, ends)))

    def ta_count_singlesegment(self, start: PosOneIdx, end: PosOneIdx) -> int:
        return self._count_substr("TA", self._chrm.seqf(start, end))

    def cg_count_singlesegment(self, start: PosOneIdx, end: PosOneIdx) -> int:
        return self._count_substr("CG", self._chrm.seqf(start, end))


class KMer(KMerParent):
    def __init__(self, chrm: Chromosome) -> None:
        super().__init__(chrm)

    def count_w_rc(
        self, kmer: KMerSeq, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> NDArray[(Any,), int]:
        return self.count(kmer, starts, ends) + self.count(
            reverse_compliment_of(kmer), starts, ends
        )

    def count(
        self, kmer: KMerSeq, starts: Iterable[PosOneIdx], ends: Iterable[PosOneIdx]
    ) -> NDArray[(Any,), int]:
        return np.array(
            list(
                map(
                    lambda se: self._count_substr(kmer, self._chrm.seqf(se[0], se[1])),
                    zip(starts, ends),
                )
            )
        )
