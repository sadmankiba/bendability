from __future__ import annotations
from typing import Iterable, NamedTuple, Any

import pandas as pd
import numpy as np
from nptyping import NDArray
from chromosome.chromosome import ChrmOperator, Chromosome
from util.util import Attr, NumpyTool
from util.constants import ONE_INDEX_START
from util.custom_types import PosOneIdx, NonNegativeInt

RegionsInternal = pd.DataFrame

START = "start"
END = "end"
MIDDLE = "middle"
MEAN_C0 = "mean_c0"


class RegionNT(NamedTuple):
    start: PosOneIdx
    end: PosOneIdx
    middle: PosOneIdx
    mean_c0: float


class Regions:
    def __init__(self, chrm: Chromosome, regions: RegionsInternal = None) -> None:
        self.chrm = chrm
        self._regions = regions if regions is not None else self._get_regions()
        self._add_middle()
        self._add_mean_c0()
        self._mean_c0 = None
        self._cvrmask = None

    def _get_regions(self) -> None:
        pass

    def __iter__(self) -> Iterable[RegionNT]:
        return self._regions.itertuples()

    def __getitem__(self, key: NonNegativeInt | str) -> pd.Series:
        if isinstance(key, NonNegativeInt):
            return self._regions.iloc[key]

        if key in self._regions.columns:
            return self._regions[key]

        raise KeyError

    def __len__(self):
        return len(self._regions)

    @property
    def mean_c0(self) -> float:
        def calc_mean_c0():
            return self.chrm.c0_spread()[self.cover_mask].mean()

        return Attr.calc_attr(self, "_mean_c0", calc_mean_c0)

    @property
    def total_bp(self) -> int:
        return self.cover_mask.sum()

    @property
    def cover_mask(self) -> NDArray[(Any,), bool]:
        def calc_cvrmask():
            return ChrmOperator(self.chrm).create_cover_mask(
                self._regions[START], self._regions[END]
            )

        return Attr.calc_attr(self, "_cvrmask", calc_cvrmask)

    def is_in_regions(
        self, bps: Iterable[PosOneIdx]
    ) -> NDArray[[Any,], bool]:
        return np.array([self.cover_mask[bp - 1] for bp in bps])

    def cover_regions(self) -> Regions:
        starts = (
            NumpyTool.match_pattern(self.cover_mask, [False, True])
            + 1
            + ONE_INDEX_START
        )
        ends = NumpyTool.match_pattern(self.cover_mask, [True, False]) + ONE_INDEX_START
        assert len(starts) == len(ends)

        rgns = Regions(self.chrm, pd.DataFrame({START: starts, END: ends}))
        return rgns

    def contains(self, bps: Iterable[PosOneIdx]) -> NDArray[(Any,), bool]:
        def _contains_bps(region: RegionNT) -> bool:
            cntns = False
            for bp in bps:
                if getattr(region, START) <= bp <= getattr(region, END):
                    cntns = True
                    break

            return cntns

        return np.array(list(map(lambda rgn: _contains_bps(rgn), self)))

    def _add_middle(self) -> None:
        self._regions[MIDDLE] = (
            (self._regions[START] + self._regions[END]) / 2
        ).astype(int)

    def _add_mean_c0(self) -> None:
        self._regions = self._regions.assign(
            mean_c0=lambda rgns: ChrmOperator(self.chrm).mean_c0_regions_indiv(
                rgns[START], rgns[END]
            )
        )


class RegionsContain:
    @classmethod
    def is_in(
        self, bps: Iterable[PosOneIdx], containers: RegionsInternal[START:int, END:int]
    ):
        pass
