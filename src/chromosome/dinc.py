from __future__ import annotations
from typing import Iterable

import regex as re
import pandas as pd

from chromosome.chromosome import Chromosome
from util.custom_types import PosOneIdx, DNASeq

DincStr = str

class Dinc:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
    
    @classmethod
    def find_pos(cls, seq: DNASeq, dincs: Iterable[DincStr]) -> dict[DincStr, list[int]]:
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

    def ta_count_singlesegment(self, start: PosOneIdx, end: PosOneIdx) -> list:
        return self._count_substr("TA", self._chrmseq(start, end))

    def cg_count_singlesegment(self, start: PosOneIdx, end: PosOneIdx) -> list:
        return self._count_substr("CG", self._chrmseq(start, end))

    def _chrmseq(self, start: PosOneIdx, end: PosOneIdx) -> str:
        return self._chrm.seq[start - 1 : end]

    def _count_substr(self, substr: str, cntnr: str) -> int:
        return len(re.findall(substr, cntnr, overlapped=True))
