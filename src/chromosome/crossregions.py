from __future__ import annotations
import random
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
from cairosvg import svg2png
from nptyping import NDArray

from chromosome.chromosome import PlotChrm
from chromosome.dinc import Dinc
from chromosome.genes import Genes
from motif.motifs import Motifs
from util.constants import FigSubDir, ONE_INDEX_START
from .chromosome import Chromosome
from chromosome.regions import END, MIDDLE, START
from .genes import Promoters, STRAND
from .nucleosomes import Linkers, Nucleosomes
from conformation.domains import BndParmT, BoundariesHE, SCORE, BndParm
from .regions import LEN, MEAN_C0, Regions
from util.util import Attr, PathObtain, PlotUtil, FileSave
from util.custom_types import PosOneIdx


class SubRegions:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        self.bnd_parm = BndParmT()
        self._prmtrs = None
        self._bndrs = None

    @property
    def bndrs(self) -> BoundariesHE:
        def _bndrs():
            return BoundariesHE(self._chrm, **self.bnd_parm)

        return Attr.calc_attr(self, "_bndrs", _bndrs)

    @property
    def prmtrs(self) -> Promoters:
        def _prmtrs():
            return Promoters(self._chrm)

        return Attr.calc_attr(self, "_prmtrs", _prmtrs)

    @property
    def genes(self) -> Genes:
        def _genes():
            return Genes(self._chrm)

        return Attr.calc_attr(self, "_genes", _genes)

    def prmtrs_with_bndrs(self):
        return self.prmtrs.with_loc(self.bndrs[MIDDLE], True)

    def prmtrs_wo_bndrs(self):
        return self.prmtrs.with_loc(self._bndrs[MIDDLE], False)


class DistribPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

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

    def box_mean_c0_bndrs_prmtrs(self) -> Path:
        sr = SubRegions(self._chrm)
        sr.bnd_parm = BndParm.HIRS_SHR
        distribs = [
            sr.bndrs[MEAN_C0],
            sr.prmtrs[MEAN_C0],
            sr.genes[MEAN_C0],
            sr.bndrs.prmtr_bndrs()[MEAN_C0],
            sr.bndrs.non_prmtr_bndrs()[MEAN_C0],
            sr.prmtrs_with_bndrs()[MEAN_C0],
            sr.prmtrs_wo_bndrs()[MEAN_C0],
        ]
        labels = [
            "bndrs",
            "prmtrs",
            "genes",
            "p bndrs",
            "np bndrs",
            "prmtrs w b",
            "prmtrs wo b",
        ]
        PlotUtil.show_grid(which="both")
        plt.boxplot(distribs, showfliers=False)
        plt.xticks(ticks=range(1, 8), labels=labels)
        plt.ylabel("Mean C0")
        plt.title("Mean C0 distrib of comb of prmtrs and bndrs")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/box_bndrs_prmtrs_{sr.bndrs}_{sr.prmtrs}.png"
        )

    def prob_distrib_mean_c0_bndrs_prmtrs(self):
        bndrs = BoundariesHE(self._chrm, **BndParm.HIRS_SHR)
        prmtrs = Promoters(self._chrm)
        prmtrs_with_bndrs = prmtrs.with_loc(bndrs[MIDDLE], True)
        prmtrs_wo_bndrs = prmtrs.with_loc(bndrs[MIDDLE], False)

        distribs = [
            bndrs[MEAN_C0],
            prmtrs[MEAN_C0],
            bndrs.prmtr_bndrs()[MEAN_C0],
            bndrs.non_prmtr_bndrs()[MEAN_C0],
            prmtrs_with_bndrs[MEAN_C0],
            prmtrs_wo_bndrs[MEAN_C0],
        ]
        labels = [
            "boundaries",
            "promoters",
            "prm boundaries",
            "nonprm boundaries",
            "promoters with bndry",
            "promoters w/o bndry",
        ]

        PlotUtil.clearfig()
        PlotUtil.show_grid()

        for d, l in zip(distribs, labels):
            PlotUtil.prob_distrib(d, l)

        plt.legend()
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/bndrs_prmtrs_prob_distrib_c0_{bndrs}"
            f"_{prmtrs}_{self._chrm.id}.png"
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


class LineC0Plot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0_bndrs_indiv_toppings(self) -> None:
        bndrsall = BoundariesHE(self._chrm, **BndParm.HIRS_WD)
        for bndrs, pstr in zip(
            [bndrsall.prmtr_bndrs(), bndrsall.non_prmtr_bndrs()], ["prmtr", "nonprmtr"]
        ):
            for bndry in bndrs:
                self._line_c0_bndry_indiv_toppings(bndry, bndrs.res, pstr)

    def _line_c0_bndry_indiv_toppings(
        self, bndry: pd.Series, res: int, pstr: str
    ) -> Path:
        self.line_c0_toppings(getattr(bndry, START), getattr(bndry, END), save=False)
        plt.title(
            f"C0 around {pstr} boundary at {getattr(bndry, MIDDLE)} bp of chrm {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm.id}/"
            f"bndry_{pstr}_{getattr(bndry, START)}_{getattr(bndry, END)}_score_"
            f"{round(getattr(bndry, SCORE), 2)}_res_{res}.png"
        )

    def line_c0_prmtrs_indiv_toppings(self) -> None:
        prmtrs = Promoters(self._chrm)
        for prmtr in prmtrs:
            self._line_c0_prmtr_indiv_toppings(prmtr)

    def _line_c0_prmtr_indiv_toppings(self, prmtr: pd.Series) -> Path:
        self.line_c0_toppings(getattr(prmtr, START), getattr(prmtr, END), save=False)
        fr = "frw" if getattr(prmtr, STRAND) == 1 else "rvs"
        plt.title(f"C0 in {fr} promoter {getattr(prmtr, START)}-{getattr(prmtr, END)}")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.PROMOTERS}/{self._chrm.id}/"
            f"{fr}_{getattr(prmtr, START)}_{getattr(prmtr, END)}.png"
        )

    def line_c0_toppings(
        self, start: PosOneIdx, end: PosOneIdx, show: bool = False, save: bool = True
    ) -> Path:
        def _within_bool(pos: pd.Series) -> pd.Series:
            return (start <= pos) & (pos <= end)

        def _within(pos: pd.Series) -> pd.Series:
            return pos.loc[(start <= pos) & (pos <= end)]

        def _end_within(rgns: Regions) -> Regions:
            return rgns[(_within_bool(rgns[START])) | (_within_bool(rgns[END]))]

        def _clip(starts: pd.Series, ends: pd.Series) -> tuple[pd.Series, pd.Series]:
            scp = starts.copy()
            ecp = ends.copy()
            scp.loc[scp < start] = start
            ecp.loc[ecp > end] = end
            return scp, ecp

        def _vertline(pos: pd.Series, color: str) -> None:
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
                points = [
                    [m - wb * s, gnd + hb * s],
                    [m, gnd],
                    [m + wb * s, gnd + hb * s],
                ]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=2)
                plt.gca().add_patch(line)

        def _lng_linkrs(lnks: Linkers, clr: str) -> None:
            sc, ec = _clip(lnks[START], lnks[END])
            for s, e in zip(sc, ec):
                rectangle = plt.Rectangle(
                    (s, gnd - 0.05), e - s, 0.1, fc=clr, alpha=0.5
                )
                plt.gca().add_patch(rectangle)

        def _tss(tss: pd.Series, frw: bool, clr: str) -> None:
            diru = 1 if frw else -1
            for t in tss:
                points = [[t, gnd], [t, gnd + 0.15], [t + 50 * diru, gnd + 0.15]]
                line = plt.Polygon(points, closed=None, fill=None, edgecolor=clr, lw=3)
                plt.gca().add_patch(line)

        def _text() -> None:
            for x, y in self._text_pos_calc(start, end, 0.1):
                plt.text(x, gnd + y, self._chrm.seq[x - 1 : x + 3], fontsize="xx-small")

        PlotUtil.clearfig()
        PlotUtil.show_grid(which="both")
        pltchrm = PlotChrm(self._chrm)
        pltchrm.line_c0(start, end)
        dyads = Nucleosomes(self._chrm)[MIDDLE]
        genes = Genes(self._chrm)
        bndrs = BoundariesHE(self._chrm, res=200, score_perc=0.5)
        lng_lnkers = Linkers(self._chrm).ndrs(40)

        colors = ["tab:orange", "tab:brown", "tab:purple", "tab:green"]
        labels = ["nuc", "bndrs", "tss", "lng lnk"]
        _nuc_ellipse(_within(dyads), colors[0])
        _bndrs(_within(bndrs[MIDDLE]), bndrs[SCORE], colors[1])
        _tss(_within(genes.frwrd_genes()[START]), True, colors[2])
        _tss(_within(genes.rvrs_genes()[END]), False, colors[2])
        _lng_linkrs(_end_within(lng_lnkers), colors[3])
        _text()

        PlotUtil.legend_custom(colors, labels)

        if show:
            plt.show()

        if save:
            return FileSave.figure_in_figdir(
                f"{FigSubDir.CROSSREGIONS}/line_c0_toppings.png"
            )

    def _text_pos_calc(
        self, start: PosOneIdx, end: PosOneIdx, amp: float
    ) -> Iterator[tuple[PosOneIdx, float]]:
        return zip(
            range(start, end, 4),
            [amp, amp / 2, -amp / 2, -amp] * math.ceil((end - start) / 4 / 4),
        )


class PlotPrmtrsBndrs:
    WB_DIR = "with_boundaries"
    WOB_DIR = "without_boundaries"
    BOTH_DIR = "both"

    def __init__(self):
        pass

    def dinc_explain(self) -> Path:
        def _total_dinc(rgns: Regions, cnt_fnc: Callable) -> NDArray[(Any,), float]:
            return np.array(cnt_fnc(rgns[START], rgns[END]))

        def _mean_dinc(rgns: Regions, cnt_fnc: Callable) -> NDArray[(Any,), float]:
            return _total_dinc(rgns, cnt_fnc) / rgns[LEN].to_numpy()

        subr = SubRegions(Chromosome("VL"))
        dinc = Dinc(Chromosome("VL"))
        pmwb = subr.prmtrs_with_bndrs()
        pmob = subr.prmtrs_wo_bndrs()
        labels = ["Prmtrs w b", "Prmtrs wo b"]
        fig, axs = plt.subplots(2)
        fig.suptitle("TpA and CpG content in promoters")
        
        for dinc, cnt_fnc, axes in zip(
            ["TpA", "CpG"],
            [dinc.ta_count_multisegment, dinc.cg_count_multisegment],
            axs,
        ):
            PlotUtil.box_many(
                [
                    _total_dinc(pmwb, cnt_fnc),
                    _total_dinc(pmob, cnt_fnc),
                ],
                labels=labels,
                ylabel=f"{dinc} content",
                pltobj=axes,
            )

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/dinc_explain_VL.png"
        )

    def both_sorted_motif_contrib(self):
        for i, num in enumerate(Motifs().sorted_contrib()):
            self._both_motif_contrib_single("both_sorted_motif", num, i)

    def both_motif_contrib(self):
        for num in range(256):
            self._both_motif_contrib_single(self.BOTH_DIR, num)

    def _both_motif_contrib_single(
        self, bthdir: str, num: int, srtidx: int = None
    ) -> Path:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(plt.imread(self._contrib_file(self.WB_DIR, num, "png")))
        axs[0].set(title="with boundaries")
        axs[1].imshow(plt.imread(self._contrib_file(self.WOB_DIR, num, "png")))
        axs[1].set(title="without boundaries")
        fig.suptitle(f"Contrib of motif {num} in promoters")
        return FileSave.figure(
            self._contrib_file(
                bthdir, f"{srtidx}_{num}" if srtidx is not None else num, "png"
            )
        )

    def _contrib_file(
        self, dir: str, num: int | str, frmt: Literal["svg"] | Literal["png"]
    ) -> str:
        return (
            f"{PathObtain.figure_dir()}/{FigSubDir.PROMOTERS}/"
            f"distribution_around_promoters/{dir}/motif_{num}.{frmt}"
        )

    def svg2png_contrib(self):
        for i in range(256):
            svg2png(
                url=self._contrib_file(self.WOB_DIR, i, "svg"),
                write_to=self._contrib_file(self.WOB_DIR, i, "png"),
            )
