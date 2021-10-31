from __future__ import annotations
from pathlib import Path
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chromosome import Chromosome
from .nucleosome import Nucleosome
from util.reader import GeneReader
from util.util import FileSave, PlotUtil, PathObtain, Attr
from util.custom_types import NonNegativeInt


class GeneNT(NamedTuple):
    """
    Representation of position of a gene.

    'start' is lower bp, 'end' is higher bp irrespective of 'strand'.
    """

    start: int
    end: int
    strand: int
    dyads: np.ndarray


START = "start"
END = "end"
STRAND = "strand"
DYADS = "dyads"


class Genes:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._genes = GeneReader().read_transcription_regions_of(chrm.number)
        self._add_dyads()

    def __iter__(self) -> Iterable[GeneNT]:
        return self._genes.itertuples()

    def __getitem__(self, key: NonNegativeInt | str) -> pd.Series:
        if isinstance(key, NonNegativeInt):
            return self._genes.iloc[key]

        if key in self._genes.columns:
            return self._genes[key]

        raise KeyError

    def _add_dyads(self) -> None:
        nucs = Nucleosome(self._chrm)
        # TODO: Nucs need not know about strand
        self._genes[DYADS] = self._genes.apply(
            lambda tr: nucs.dyads_between(tr[START], tr[END], tr[STRAND]), axis=1
        )

    def frwrd_genes(self) -> Genes:
        genes = Genes(self._chrm)
        genes._genes = self._genes.query(f"{STRAND} == 1").copy()
        return genes

    def rvrs_genes(self) -> Genes:
        genes = Genes(self._chrm)
        genes._genes = self._genes.query(f"{STRAND} == -1").copy()
        return genes 

    def plot_mean_c0_vs_dist_from_dyad(self) -> Path:
        frwrd_p1_dyads = self.frwrd_genes()[DYADS].apply(lambda dyads: dyads[0])
        frwrd_mean_c0 = self._chrm.mean_c0_around_bps(frwrd_p1_dyads, 600, 400)

        rvrs_p1_dyads = self.rvrs_genes()[DYADS].apply(lambda dyads: dyads[0])
        rvrs_mean_c0 = self._chrm.mean_c0_around_bps(rvrs_p1_dyads, 400, 600)[::-1]

        mean_c0 = (
            frwrd_mean_c0 * len(frwrd_p1_dyads) + rvrs_mean_c0 * len(rvrs_p1_dyads)
        ) / len(self._genes)

        plt.close()
        plt.clf()
        PlotUtil.show_grid()
        plt.plot(np.arange(-600, 400 + 1), mean_c0)

        plt.xlabel("Distance from dyad (bp)")
        plt.ylabel("Mean C0")
        plt.title(
            f"{self._chrm.c0_type} Mean C0 around +1 dyad"
            f" in chromosome {self._chrm.number}"
        )

        return FileSave.figure(
            f"{PathObtain.figure_dir()}/gene/dist_p1_dyad_{self._chrm}.png"
        )

    def in_promoter(self, bps: np.ndarray | list[int] | pd.Series) -> np.ndarray:
        """
        Find whether some bps lies in promoter

        Promoter is defined as +-400bp from TSS
        """
        frwrd_prmtr_rgn = self._frwrd_tr_df()[START].apply(
            lambda tss: np.arange(tss - 400, tss + 400 + 1)
        )
        rvrs_prmtr_rgn = self._rvrs_tr_df()[END].apply(
            lambda tss: np.arange(tss - 400, tss + 400 + 1)
        )

        prmtr_rgn = np.concatenate(
            (
                np.array(frwrd_prmtr_rgn.tolist()).flatten(),
                np.array(rvrs_prmtr_rgn.tolist()).flatten(),
            )
        )

        return np.array([bp in prmtr_rgn for bp in bps])


class Promoters:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self._upstream_tss = 500
        self._downstream_tss = 0
        self._promoters = self._get_promoters()

    def _get_promoters(self) -> pd.DataFrame[START:int, END:int, STRAND:int]:
        genes = Genes(self._chrm)
        return pd.DataFrame(
            {
                START: pd.concat([
                    genes.frwrd_genes()[START] - self._upstream_tss,
                    genes.rvrs_genes()[END] - self._downstream_tss,
                ]),
                END: pd.concat([
                    genes.frwrd_genes()[START] + self._downstream_tss,
                    genes.rvrs_genes()[END] + self._upstream_tss,
                ]),
                STRAND: pd.concat([
                    genes.frwrd_genes()[STRAND], genes.rvrs_genes()[STRAND]
                ]),
            }
        )

    @property
    def mean_c0(self):
        def calc_mean_c0():
            return self._chrm.c0_spread()[self.cover_mask()].mean()

        return Attr.calc_attr(self, "_mean_c0", calc_mean_c0)

    def cover_mask(self) -> np.ndarray:
        return self._chrm.get_cvr_mask(
            self._promoters[START], 0, self._upstream_tss + self._downstream_tss
        )
