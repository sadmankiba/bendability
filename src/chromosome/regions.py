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
LEN = "len"
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
        self._add_len()
        self._add_middle()
        self._add_mean_c0()
        self._mean_c0 = None
        self._cvrmask = None

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

    def __sub__(self, other: Regions):
        result_rgns = self._regions[
            ~(
                self._regions[START].isin(other._regions[START])
                & self._regions[END].isin(other._regions[END])
            )
        ]
        return self._new(result_rgns)

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

    def with_rgn(self, rgns: Regions) -> Regions:
        cntns = self._contains_rgn(rgns)
        return self._new(self._regions.iloc[cntns])

    def _contains_rgn(self, rgns: Regions) -> NDArray[(Any,), bool]:
        def _rgn_contains_rgn(cntn: RegionNT) -> bool:
            cntns = False
            for rgn in rgns:
                if getattr(cntn, START) <= getattr(rgn, START) and getattr(
                    rgn, END
                ) <= getattr(cntn, END):
                    cntns = True
                    break
            return cntns

        return np.array(list(map(lambda cntn: _rgn_contains_rgn(cntn), self)))

    def with_loc(self, locs: Iterable[PosOneIdx], with_x: bool) -> Regions:
        cntns = self._contains_loc(locs)
        return self._new(self._regions.iloc[cntns if with_x else ~cntns])

    def _contains_loc(self, locs: Iterable[PosOneIdx]) -> NDArray[(Any,), bool]:
        def _rgn_contains_loc(region: RegionNT) -> bool:
            cntns = False
            for loc in locs:
                if getattr(region, START) <= loc <= getattr(region, END):
                    cntns = True
                    break

            return cntns

        return np.array(list(map(lambda rgn: _rgn_contains_loc(rgn), self)))

    def rgns_contained_in(self, containers: Regions) -> Regions:
        cntnd = self.contained_in(containers)
        return self._new(self._regions.iloc[cntnd])

    def contained_in(self, containers: Regions) -> NDArray[(Any,), bool]:
        def _in_containers(rgn: RegionNT):
            inc = False
            for cntn in containers:
                if getattr(cntn, START) <= getattr(rgn, START) and getattr(
                    rgn, END
                ) <= getattr(cntn, END):
                    inc = True
                    break

            return inc

        return np.array(list(map(lambda rgn: _in_containers(rgn), self)))

    def len_at_least(self, len: int) -> Regions:
        return self._new(self._regions.query(f"{LEN} >= {len}"))

    def _get_regions(self) -> RegionsInternal:
        pass

    def _new(self, rgns: RegionsInternal) -> Regions:
        return type(self)(self.chrm, rgns)

    def _add_len(self) -> None:
        self._regions[LEN] = self._regions[END] - self._regions[START] + 1

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
