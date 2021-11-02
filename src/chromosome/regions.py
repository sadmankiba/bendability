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


class RegionNT(NamedTuple):
    start: PosOneIdx
    end: PosOneIdx


class Regions:
    def __init__(self, chrm: Chromosome) -> None:
        self.chrm = chrm
        self._regions = self._get_regions()
        self._mean_c0 = None 
        self._cvrmask = None

    def _get_regions(self) -> RegionsInternal:
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

    def cover_regions(self) -> Regions:
        starts = (
            NumpyTool.match_pattern(self.cover_mask, [False, True]) + 1 + ONE_INDEX_START
        )
        ends = (
            NumpyTool.match_pattern(self.cover_mask, [True, False]) + ONE_INDEX_START
        )
        assert len(starts) == len(ends)

        rgns = Regions(self.chrm)
        rgns._regions = self.with_middle(pd.DataFrame({START: starts, END: ends}))
        return rgns
    
    @classmethod
    def with_middle(self, rgns: RegionsInternal) -> RegionsInternal:
        rgns[MIDDLE] = ((rgns[START] + rgns[END]) / 2).astype(int)
        return rgns


class RegionsContain:
    @classmethod
    def is_in(
        self, bps: Iterable[PosOneIdx], containers: RegionsInternal[START:int, END:int]
    ):
        pass

    @classmethod
    def contains(
        self,
        containers: RegionsInternal[START:PosOneIdx, END:PosOneIdx],
        bps: Iterable[PosOneIdx],
    ) -> NDArray[(Any,), bool]:
        def _contains_bps(region: NamedTuple[START:PosOneIdx, END:PosOneIdx]):
            cntns = False
            for bp in bps:
                if getattr(region, START) <= bp <= getattr(region, END):
                    cntns = True

            return cntns

        return np.array(list(map(lambda cnt: _contains_bps(cnt), containers)))
