import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chromosome.chromosome import PlotChrm
from chromosome.genes import Genes
from util.constants import FigSubDir, ONE_INDEX_START
from .chromosome import Chromosome
from chromosome.regions import END, MIDDLE, START
from .genes import Promoters
from .nucleosomes import Linkers, NUC_HALF, Nucleosomes
from conformation.domains import BoundariesHE
from .regions import LEN, MEAN_C0, Regions
from util.util import PlotUtil, FileSave
from util.custom_types import PosOneIdx


class CrossRegionsPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0_toppings(
        self, start: PosOneIdx, end: PosOneIdx, show: bool = False
    ) -> Path:
        def _within(pos: pd.Series) -> pd.Series:
            return pos.loc[(start <= pos) & (pos <= end)]

        def _topping(pos: pd.Series, color: str) -> None:
            for p in _within(pos):
                PlotUtil.vertline(p, color=color)

        PlotUtil.clearfig()
        PlotUtil.show_grid(which="minor")
        pltchrm = PlotChrm(self._chrm)
        pltchrm.line_c0(start, end)
        dyads = Nucleosomes(self._chrm)[MIDDLE]
        genes = Genes(self._chrm)

        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
        ]
        labels = ["dyad", "-73bp", "+73bp", "bndrs", "frw tss", "rvs tss"]
        _topping(dyads, colors[0])
        _topping(dyads - NUC_HALF, colors[1])
        _topping(dyads + NUC_HALF, colors[2])
        _topping(BoundariesHE(self._chrm, res=200, score_perc=0.5)[MIDDLE], colors[3])
        _topping(genes.frwrd_genes()[START], colors[4])
        _topping(genes.rvrs_genes()[END], colors[5])
        PlotUtil.legend_custom(colors, labels)

        if show:
            plt.show()

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/line_c0_toppings.png"
        )

    def distrib_cuml_bndrs_nearest_tss_distnc(self) -> Path:
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)
        self._distrib_cuml_nearest_tss_distnc(bndrs)
        plt.title(
            f"Cumulative perc. of distance from boundary res={bndrs.res} bp "
            f"middle to nearest TSS"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_tss_distrib_cuml_res_{bndrs.res}_"
            f"perc_{bndrs.score_perc}"
            f"_{self._chrm.number}.png"
        )

    def distrib_cuml_random_locs_nearest_tss_distnc(self) -> Path:
        rndlocs = [
            random.randint(ONE_INDEX_START, self._chrm.total_bp) for _ in range(1000)
        ]
        locs = BoundariesHE(
            self._chrm, regions=pd.DataFrame({START: rndlocs, END: rndlocs})
        )
        self._distrib_cuml_nearest_tss_distnc(locs)
        plt.title(f"Cumulative perc. of distance from random pos to nearest TSS")
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_tss_random_pos_distrib_cuml_{self._chrm.number}.png"
        )

    def _distrib_cuml_nearest_tss_distnc(self, bndrs: BoundariesHE):
        genes = Genes(self._chrm)
        distns = bndrs.nearest_locs_distnc(
            pd.concat([genes.frwrd_genes()[START], genes.rvrs_genes()[END]])
        )

        PlotUtil.distrib_cuml(distns)
        plt.xlim(-1000, 1000)
        PlotUtil.show_grid()
        plt.xlabel("Distance")
        plt.ylabel("Percentage")

    def distrib_cuml_bndrs_nearest_ndr_distnc(
        self, min_lnker_len: list[int] = [80, 60, 40, 30]
    ) -> Path:
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)
        self._distrib_cuml_nearest_ndr_distnc(bndrs, min_lnker_len)
        plt.title(
            f"Cumulative perc. of distance from boundary res={bndrs.res} bp "
            f"middle to nearest NDR >= x bp"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_ndr_distrib_cuml_res_{bndrs.res}_"
            f"perc_{bndrs.score_perc}_{'_'.join(str(i) for i in min_lnker_len)}"
            f"_{self._chrm.number}.png"
        )

    def distrib_cuml_random_locs_nearest_ndr_distnc(
        self, min_lnker_len: list[int] = [80, 60, 40, 30]
    ) -> Path:
        rndlocs = [
            random.randint(ONE_INDEX_START, self._chrm.total_bp) for _ in range(1000)
        ]
        locs = BoundariesHE(
            self._chrm, regions=pd.DataFrame({START: rndlocs, END: rndlocs})
        )
        self._distrib_cuml_nearest_ndr_distnc(locs, min_lnker_len)
        plt.title(
            f"Cumulative perc. of distance from random pos" f" to nearest NDR >= x bp"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_ndr_random_pos_distrib_cuml"
            f"_{'_'.join(str(i) for i in min_lnker_len)}"
            f"_{self._chrm.number}.png"
        )

    def _distrib_cuml_nearest_ndr_distnc(
        self, bndrs: BoundariesHE, min_lnker_len: list[int]
    ):
        lnkrs = Linkers(self._chrm)
        for llen in min_lnker_len:
            distns = bndrs.nearest_locs_distnc(lnkrs.ndrs(llen)[MIDDLE])
            PlotUtil.distrib_cuml(distns, label=str(llen))

        plt.legend()
        plt.xlim(-1000, 1000)
        PlotUtil.show_grid()
        plt.xlabel("Distance")
        plt.ylabel("Percentage")

    def prob_distrib_bndrs_nearest_ndr_distnc(
        self, min_lnker_len: list[int] = [80, 60, 40, 30]
    ) -> Path:
        bndrs = BoundariesHE(self._chrm)
        lnkrs = Linkers(self._chrm)
        for llen in min_lnker_len:
            distns = bndrs.nearest_locs_distnc(lnkrs.ndrs(llen)[MIDDLE])
            PlotUtil.prob_distrib(distns, label=str(llen))

        plt.legend()
        plt.xlim(-1000, 1000)
        PlotUtil.show_grid()
        plt.xlabel("Distance")
        plt.ylabel("Prob distrib")
        plt.title(
            f"Prob distrib of distance from boundary res={bndrs.res} bp "
            f"middle to nearest NDR >= x bp"
        )
        return FileSave.figure_in_figdir(
            f"boundaries/distnc_ndr_prob_distrib_res_{bndrs.res}_{self._chrm.number}.png"
        )

    def num_prmtrs_bndrs_ndrs(self) -> Path:
        ndrs = Linkers(self._chrm).ndrs()
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.with_rgn(ndrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        bndrs = BoundariesHE(self._chrm)
        bndrs_with_ndr = bndrs.with_rgn(ndrs)
        bndrs_wo_ndr = bndrs - bndrs_with_ndr
        PlotUtil.clearfig()
        plt.bar(
            [0, 1, 3, 4],
            [
                len(prmtrs_with_ndr),
                len(prmtrs_wo_ndr),
                len(bndrs_with_ndr),
                len(bndrs_wo_ndr),
            ],
        )
        plt.xticks(
            [0, 1, 3, 4],
            [
                "Promoters with NDR",
                "Promoters without NDR",
                "Boundaries with NDR",
                "Boundaries without NDR",
            ],
        )
        plt.title(
            f"Promoters and Boundaries with and without NDR in {self._chrm.number}"
        )
        return FileSave.figure_in_figdir(
            f"genes/num_prmtrs_bndrs_ndr_{self._chrm.number}.png"
        )

    def prob_distrib_prmtr_ndrs(self) -> Path:
        lng_lnkrs = Linkers(self._chrm).ndrs()
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.with_rgn(lng_lnkrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        PlotUtil.clearfig()
        PlotUtil.show_grid()
        PlotUtil.prob_distrib(prmtrs[MEAN_C0], "Promoters")
        PlotUtil.prob_distrib(prmtrs_with_ndr[MEAN_C0], "Promoters with NDR")
        PlotUtil.prob_distrib(prmtrs_wo_ndr[MEAN_C0], "Promoters without NDR")
        plt.xlabel("Mean C0")
        plt.ylabel("Prob Distribution")
        plt.legend()
        return FileSave.figure_in_figdir("genes/prob_distrib_prmtr_ndrs.png")

    def prob_distrib_linkers_len_in_prmtrs(self) -> Path:
        PlotUtil.clearfig()
        PlotUtil.show_grid()
        linkers = Linkers(self._chrm)
        PlotUtil.prob_distrib(linkers[LEN], "linkers")
        prmtrs = Promoters(linkers.chrm)
        PlotUtil.prob_distrib(linkers.rgns_contained_in(prmtrs)[LEN], "prm linkers")
        plt.legend()
        plt.xlabel("Length")
        plt.ylabel("Prob distribution")
        plt.xlim(0, 300)
        return FileSave.figure_in_figdir(
            f"linkers/prob_distr_len_prmtrs_{self._chrm.number}.png"
        )
