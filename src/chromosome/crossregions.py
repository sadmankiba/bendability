import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd

from chromosome.chromosome import PlotChrm
from chromosome.genes import Genes
from util.constants import FigSubDir, ONE_INDEX_START
from .chromosome import Chromosome
from chromosome.regions import END, MIDDLE, START
from .genes import Promoters, STRAND
from .nucleosomes import Linkers, NUC_HALF, Nucleosomes
from conformation.domains import BoundariesHE, SCORE
from .regions import LEN, MEAN_C0
from util.util import PlotUtil, FileSave
from util.custom_types import PosOneIdx


class CrossRegionsPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0_prmtrs_indiv_toppings(self) -> None:
        prmtrs = Promoters(self._chrm)
        for prmtr in prmtrs:
            self.line_c0_toppings(getattr(prmtr, START), getattr(prmtr, END), save=False)
            fr = "frw" if getattr(prmtr, STRAND) == 1 else "rvs"
            plt.title(
                f"C0 in {fr} promoter {getattr(prmtr, START)}-{getattr(prmtr, END)}"
            )
            FileSave.figure_in_figdir(
                f"{FigSubDir.PROMOTERS}/{self._chrm.id}/"
                f"{fr}_{getattr(prmtr, START)}_{getattr(prmtr, END)}.png"
            )

    def line_c0_toppings(
        self, start: PosOneIdx, end: PosOneIdx, show: bool = False, save: bool = True
    ) -> Path:
        def _within(pos: pd.Series) -> pd.Series:
            return pos.loc[(start <= pos) & (pos <= end)]

        def _topping(pos: pd.Series, color: str) -> None:
            for p in _within(pos):
                PlotUtil.vertline(p, color=color)
        
        gnd = self._chrm.mean_c0
        def _nuc_ellipse(dyads: pd.Series, clr: str) -> None:
            for d in dyads:
                ellipse = Ellipse(
                    xy=(d, gnd),
                    width=146,
                    height=0.2,
                    edgecolor=clr,
                    fc="None",
                    lw=1,
                )
                plt.gca().add_patch(ellipse)
        
        def _bndrs(mids: pd.Series, scr: pd.Series, clr: str) -> None: 
            strngth = 1 - scr
            wb = 150
            hb = 0.1
            for m, s in zip(mids, strngth): 
                points = [[m - wb * s, gnd + hb * s], [m, gnd], [m + wb * s, gnd + hb * s]]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=2)
                plt.gca().add_patch(line)

        def _tss(tss: pd.Series, frw: bool, clr: str) -> None:
            diru = 1 if frw else -1
            for t in tss:
                points = [[t, gnd], [t, gnd + 0.15], [t + 50 * diru, gnd + 0.15]]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=3)
                plt.gca().add_patch(line)
        
        PlotUtil.clearfig()
        PlotUtil.show_grid(which="both")
        pltchrm = PlotChrm(self._chrm)
        pltchrm.line_c0(start, end)
        dyads = Nucleosomes(self._chrm)[MIDDLE]
        genes = Genes(self._chrm)
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)

        colors = [
            "tab:orange",
            "tab:brown",
            "tab:purple",
        ]
        labels = ["dyad", "bndrs", "tss"]
        _nuc_ellipse(_within(dyads), colors[0])
        _bndrs(_within(bndrs[MIDDLE]), bndrs[SCORE], colors[1])
        _tss(_within(genes.frwrd_genes()[START]), True, colors[2])
        _tss(_within(genes.rvrs_genes()[END]), False, colors[2])

        PlotUtil.legend_custom(colors, labels)

        if show:
            plt.show()

        if save:
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
        min_lnkr_len = 40
        ndrs = Linkers(self._chrm).ndrs(min_lnkr_len)
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.with_rgn(ndrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        bndrs = BoundariesHE(self._chrm, res=200)
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
            f"genes/num_prmtrs_bndrs_{bndrs}_ndr_{min_lnkr_len}_{self._chrm.number}.png"
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
