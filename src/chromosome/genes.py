from __future__ import annotations
import random
from pathlib import Path
from typing import Iterable, NamedTuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nptyping import NDArray

from .chromosome import Chromosome
from .nucleosomes import Nucleosomes
from .regions import Regions, RegionsContain, RegionsInternal
from util.reader import GeneReader
from util.util import FileSave, PlotUtil, PathObtain, Attr
from util.custom_types import NonNegativeInt, PosOneIdx

START = "start"
END = "end"
STRAND = "strand"
DYADS = "dyads"
MEAN_C0 = "mean_c0"


class GeneNT(NamedTuple):
    """
    Representation of a gene region.

    'start' is lower bp, 'end' is higher bp irrespective of 'strand'.
    """

    start: int
    end: int
    strand: int
    dyads: np.ndarray
    mean_c0: float


class Genes:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm
        self._genes = GeneReader().read_transcription_regions_of(chrm.number)
        self._add_dyads()
        self._add_mean_c0()

    def __iter__(self) -> Iterable[GeneNT]:
        return self._genes.itertuples()

    def __getitem__(self, key: NonNegativeInt | str) -> pd.Series:
        if isinstance(key, NonNegativeInt):
            return self._genes.iloc[key]

        if key in self._genes.columns:
            return self._genes[key]

        raise KeyError

    def _add_dyads(self) -> None:
        nucs = Nucleosomes(self._chrm)
        # TODO: Nucs need not know about strand
        self._genes[DYADS] = self._genes.apply(
            lambda tr: nucs.dyads_between(tr[START], tr[END], tr[STRAND]), axis=1
        )

    def _add_mean_c0(self) -> None:
        self._genes[MEAN_C0] = self._genes.apply(
            lambda gene: self._chrm.mean_c0_segment(gene[START], gene[END]), axis=1
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
            f"{PathObtain.figure_dir()}/genes/dist_p1_dyad_{self._chrm}.png"
        )


class Promoters(Regions):
    def __init__(
        self,
        chrm: Chromosome,
        ustr_tss: int = 500,
        dstr_tss: int = -1,
        regions: RegionsInternal = None,
    ) -> None:
        self._ustr_tss = ustr_tss
        self._dstr_tss = dstr_tss
        super().__init__(chrm, regions)

    def _get_regions(self) -> pd.DataFrame[START:int, END:int, STRAND:int]:
        genes = Genes(self.chrm)
        return pd.DataFrame(
            {
                START: pd.concat(
                    [
                        genes.frwrd_genes()[START] - self._ustr_tss,
                        genes.rvrs_genes()[END] - self._dstr_tss,
                    ],
                    ignore_index=True,
                ),
                END: pd.concat(
                    [
                        genes.frwrd_genes()[START] + self._dstr_tss,
                        genes.rvrs_genes()[END] + self._ustr_tss,
                    ],
                    ignore_index=True,
                ),
                STRAND: pd.concat(
                    [genes.frwrd_genes()[STRAND], genes.rvrs_genes()[STRAND]],
                    ignore_index=True,
                ),
            }
        )

    def __str__(self) -> str:
        return f"ustr_{self._ustr_tss}_dstr_{self._dstr_tss}"

    def and_x(self, bps: Iterable[PosOneIdx], with_x: bool):
        cntns = self.contains(bps)
        prmtrs = Promoters(
            self.chrm,
            self._ustr_tss,
            self._dstr_tss,
            self._regions.iloc[cntns if with_x else ~cntns],
        )
        return prmtrs


class PromotersPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._prmtrs = Promoters(chrm)

    def prob_distrib_c0(self) -> Path:
        sns.distplot(self._prmtrs[MEAN_C0], hist=False, kde=True)
        return FileSave.figure_in_figdir(
            f"genes/promoters_prob_distrib_c0_{self._prmtrs}_{self._prmtrs.chrm}.png"
        )

    def hist_c0(self) -> Path:
        PlotUtil.clearfig()
        plt.hist(self._prmtrs[MEAN_C0])
        return FileSave.figure_in_figdir(
            f"genes/promoters_hist_c0_{self._prmtrs}_{self._prmtrs.chrm}.png"
        )
