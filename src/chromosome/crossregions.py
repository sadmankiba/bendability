from __future__ import annotations
import random
import math
from pathlib import Path
from enum import Enum, auto
from typing import Any, Callable, Iterator, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.ticker as plticker
import pandas as pd
import numpy as np
from cairosvg import svg2png
from nptyping import NDArray

from chromosome.chromosome import PlotChrm
from chromosome.dinc import Dinc
from chromosome.genes import Genes
from motif.motifs import MotifsM30
from util.constants import FigSubDir, ONE_INDEX_START

from chromosome.dinc import KMer
from .chromosome import Chromosome
from chromosome.regions import END, MIDDLE, START
from .genes import Promoters, STRAND
from .nucleosomes import Linkers, Nucleosomes
from conformation.domains import (
    BndParmT,
    BoundariesHE,
    SCORE,
    BndParm,
    BoundariesType,
    BoundariesFactory,
    BndFParm,
    BndSel,
)
from conformation.loops import LoopAnchors, LoopInsides
from .regions import LEN, MEAN_C0, Regions
from util.util import Attr, PathObtain, PlotUtil, FileSave
from util.custom_types import PosOneIdx, KMerSeq


class SubRegions:
    def __init__(self, chrm: Chromosome) -> None:
        self.chrm = chrm
        self._prmtrs = None
        self._bndrs = None
        self.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        self.min_ndr_len = 40

    @property
    def bndrs(self) -> BoundariesHE:
        def _bndrs():
            return BoundariesFactory(self.chrm).get_bndrs(self.bsel)

        return Attr.calc_attr(self, "_bndrs", _bndrs)

    @property
    def prmtrs(self) -> Promoters:
        def _prmtrs():
            return Promoters(self.chrm)

        return Attr.calc_attr(self, "_prmtrs", _prmtrs)

    @property
    def genes(self) -> Genes:
        def _genes():
            return Genes(self.chrm)

        return Attr.calc_attr(self, "_genes", _genes)

    @property
    def nucs(self) -> Nucleosomes:
        def _nucs():
            return Nucleosomes(self.chrm)

        return Attr.calc_attr(self, "_nucs", _nucs)

    @property
    def lnkrs(self) -> Linkers:
        def _lnkrs():
            return Linkers(self.chrm)

        return Attr.calc_attr(self, "_lnkrs", _lnkrs)

    @property
    def ndrs(self) -> Linkers:
        def _ndrs():
            return self.lnkrs.ndrs(self.min_ndr_len)

        return Attr.calc_attr(self, "_ndrs", _ndrs)

    @property 
    def lpancrs(self):
        return LoopAnchors(self.chrm, lim=250)
    
    @property 
    def lpinsds(self):
        return LoopInsides(self.lpancrs)

    @property
    def bsel(self) -> BndSel:
        return self._bsel

    @bsel.setter
    def bsel(self, _bsel: BndSel):
        self._bsel = _bsel
        self._bndrs = None

    def prmtrs_with_bndrs(self):
        return self.prmtrs.with_loc(self.bndrs[MIDDLE], True)

    def prmtrs_wo_bndrs(self):
        return self.prmtrs.with_loc(self._bndrs[MIDDLE], False)

    def prmtr_bndrs(self):
        # TODO: Update def. in +- 100bp of promoters
        return self.bndrs.mid_contained_in(self.prmtrs)

    def non_prmtr_bndrs(self):
        return self.bndrs - self.prmtr_bndrs()

    def bndry_nucs(self) -> Nucleosomes:
        return self.nucs.overlaps_with_rgns(self.bndrs, 50)

    def non_bndry_nucs(self) -> Nucleosomes:
        return self.nucs - self.bndry_nucs()

    def bndry_ndrs(self) -> Linkers:
        return self.ndrs.overlaps_with_rgns(self.bndrs, self.min_ndr_len)

    def non_bndry_ndrs(self) -> Linkers:
        return self.ndrs - self.bndry_ndrs()


class Distrib(Enum):
    BNDRS = auto()
    BNDRS_E_100 = auto()
    BNDRS_E_N50 = auto()
    NUCS = auto()
    NUCS_B = auto()
    NUCS_NB = auto()
    LNKRS = auto()
    NDRS = auto()
    NDRS_B = auto()
    NDRS_NB = auto()
    PRMTRS = auto()
    GENES = auto()
    BNDRS_P = auto()
    BNDRS_NP = auto()
    PRMTRS_B = auto()
    PRMTRS_NB = auto()
    LPANCRS = auto()
    LPINSDS = auto()


class LabeledDistribs:
    def __init__(self, sr: SubRegions):
        self._sr = sr

    def dl(self, ds: list[Distrib]) -> list[tuple[np.ndarray, str]]:
        def _dl(d: Distrib):
            if d == Distrib.BNDRS:
                return self._sr.bndrs[MEAN_C0], "bndrs l 100"
            if d == Distrib.BNDRS_E_100:
                return self._sr.bndrs.extended(100)[MEAN_C0], "bndrs l 200"
            if d == Distrib.BNDRS_E_N50:
                return self._sr.bndrs.extended(-50)[MEAN_C0], "bndrs l 50"
            if d == Distrib.NUCS:
                return self._sr.nucs[MEAN_C0], "nucs"
            if d == Distrib.NUCS_B:
                return self._sr.bndry_nucs()[MEAN_C0], "nucs b"
            if d == Distrib.NUCS_NB:
                return self._sr.non_bndry_nucs()[MEAN_C0], "nucs nb"
            if d == Distrib.LNKRS:
                return self._sr.lnkrs[MEAN_C0], "lnkrs"
            if d == Distrib.NDRS:
                return self._sr.ndrs[MEAN_C0], "ndrs 40"
            if d == Distrib.NDRS_B:
                return self._sr.bndry_ndrs()[MEAN_C0], "ndrs b"
            if d == Distrib.NDRS_NB:
                return self._sr.non_bndry_ndrs()[MEAN_C0], "ndrs nb"
            if d == Distrib.PRMTRS:
                return self._sr.prmtrs[MEAN_C0], "prmtrs"
            if d == Distrib.GENES:
                return self._sr.genes[MEAN_C0], "genes"
            if d == Distrib.BNDRS_P:
                return self._sr.prmtr_bndrs()[MEAN_C0], "bndrs p"
            if d == Distrib.BNDRS_NP:
                return self._sr.non_prmtr_bndrs()[MEAN_C0], "bndrs np"
            if d == Distrib.PRMTRS_B:
                return self._sr.prmtrs_with_bndrs()[MEAN_C0], "prmtrs b"
            if d == Distrib.PRMTRS_NB:
                return self._sr.prmtrs_wo_bndrs()[MEAN_C0], "prmtrs nb"
            if d == Distrib.LPANCRS:
                return self._sr.lpancrs[MEAN_C0], "lp ancrs"
            if d == Distrib.LPINSDS:
                return self._sr.lpinsds[MEAN_C0], "lp insds"

        return list(map(_dl, ds))


class DistribPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def box_mean_c0_bndrs(self) -> Path:
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        sr.bsel = bsel_hexp
        ld = LabeledDistribs(sr)
        grp_bndrs_nucs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.BNDRS_E_100,
                    Distrib.BNDRS_E_N50,
                    Distrib.NUCS,
                    Distrib.NUCS_B,
                    Distrib.NUCS_NB,
                    Distrib.LNKRS,
                    Distrib.NDRS,
                    Distrib.NDRS_B,
                    Distrib.NDRS_NB,
                ]
            ),
            "title": "Mean C0 distrib of bndrs and nucs",
            "fname": f"bndrs_nucs_{sr.bndrs}.png",
        }
        grp_bndrs_prmtrs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.PRMTRS,
                    Distrib.GENES,
                    Distrib.BNDRS_P,
                    Distrib.BNDRS_NP,
                    Distrib.PRMTRS_B,
                    Distrib.PRMTRS_NB,
                ]
            ),
            "title": "Mean C0 distrib of comb of prmtrs and bndrs",
            "fname": f"bndrs_prmtrs_{sr.bndrs}_{sr.prmtrs}_{self._chrm.id}.png",
        }
        return self.box_mean_c0(grp_bndrs_nucs)
    
    @classmethod
    def box_mean_c0(cls, grp: dict):
        PlotUtil.show_grid(which="both")
        distribs = [d for d, _ in grp["dls"]]
        labels = [l for _, l in grp["dls"]]

        plt.boxplot(distribs, showfliers=False)
        plt.ylim(-0.5, 0.1)
        plt.xticks(
            ticks=range(1, len(labels) + 1),
            labels=labels,
        )
        plt.ylabel("Mean C0")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/c0_box/{grp['fname']}"
        )

    def prob_distrib_mean_c0_bndrs(self):
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_25)
        sr.bsel = bsel_hexp
        ld = LabeledDistribs(sr)
        grp_bndrs_nucs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.BNDRS_E_N50,
                    Distrib.LNKRS,
                    Distrib.NDRS,
                    Distrib.NDRS_B,
                    Distrib.NDRS_NB,
                ]
            ),
            "title": "Prob distrib of mean C0 distrib of bndrs and nucs",
            "fname": f"bndrs_nucs_{sr.bndrs}_{self._chrm.id}.png",
        }
        grp_bndrs_prmtrs = {
            "dls": ld.dl(
                [
                    Distrib.BNDRS,
                    Distrib.PRMTRS,
                    Distrib.GENES,
                    Distrib.BNDRS_P,
                    Distrib.BNDRS_NP,
                    Distrib.PRMTRS_B,
                    Distrib.PRMTRS_NB,
                ]
            ),
            "title": "Prob distrib of mean C0 distrib of comb of prmtrs and bndrs",
            "fname": f"bndrs_prmtrs_{sr.bndrs}_{sr.prmtrs}_{self._chrm.id}.png",
        }
        grp = grp_bndrs_nucs

        PlotUtil.clearfig()
        PlotUtil.show_grid()

        for d, l in grp["dls"]:
            PlotUtil.prob_distrib(d, l)

        plt.legend()
        plt.xlabel("Mean c0")
        plt.ylabel("Probability")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/prob_distrib/{grp['fname']}"
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

    def num_prmtrs_bndrs_ndrs(self, frml: int, btype: BoundariesType) -> Path:
        min_lnkr_len = 40

        def _includes(*rgnss):
            if frml == 1:
                return Regions.with_rgn(*rgnss)
            elif frml == 2:
                return Regions.overlaps_with_rgns(*rgnss, min_ovbp=min_lnkr_len)

        ndrs = Linkers(self._chrm).ndrs(min_lnkr_len)
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = _includes(prmtrs, ndrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr

        if btype == BoundariesType.HEXP:
            bparm = BndParm.HIRS_SHR
        elif btype == BoundariesType.FANC:
            bparm = BndFParm.SHR_25

        bndrs = BoundariesFactory(self._chrm).get_bndrs(BndSel(btype, bparm))
        bndrs_with_ndr = _includes(bndrs, ndrs)
        bndrs_wo_ndr = bndrs - bndrs_with_ndr
        ndrs_in_prmtrs = _includes(ndrs, prmtrs)
        ndrs_out_prmtrs = ndrs - ndrs_in_prmtrs
        ndrs_in_bndrs = _includes(ndrs, bndrs)
        ndrs_out_bndrs = ndrs - ndrs_in_bndrs
        PlotUtil.clearfig()
        PlotUtil.show_grid("major")
        plt_items = [
            (1, len(prmtrs_with_ndr), "Prm w ND"),
            (2, len(prmtrs_wo_ndr), "Prm wo ND"),
            (4, len(bndrs_with_ndr), "Bnd w ND"),
            (5, len(bndrs_wo_ndr), "Bnd wo ND"),
            (7, len(ndrs_in_prmtrs), "ND i Prm"),
            (8, len(ndrs_out_prmtrs), "ND o Prm"),
            (10, len(ndrs_in_bndrs), "ND i Bnd"),
            (11, len(ndrs_out_bndrs), "ND o Bnd"),
        ]
        plt.bar(
            list(map(lambda x: x[0], plt_items)),
            list(map(lambda x: x[1], plt_items)),
        )
        plt.xticks(
            list(map(lambda x: x[0], plt_items)),
            list(map(lambda x: x[2], plt_items)),
        )
        plt.title(
            f"Promoters and Boundaries with and without NDR in {self._chrm.number}"
        )
        figpath = FileSave.figure_in_figdir(
            f"genes/num_prmtrs_bndrs_{bndrs}_ndr_{min_lnkr_len}_{self._chrm.number}.png"
        )

        fig, ax = plt.subplots()
        # ax.plot(x,y)
        loc = plticker.MultipleLocator(
            base=50
        )  # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)

        return figpath

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


class ScatterPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def scatter_c0(self) -> Path:
        PlotUtil.clearfig()
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_25)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(
            sr.non_bndry_ndrs()[LEN],
            sr.non_bndry_ndrs()[MEAN_C0],
            c="b",
            marker="s",
            label="Non-bndry NDRs",
        )
        ax1.scatter(
            sr.bndry_ndrs()[LEN],
            sr.bndry_ndrs()[MEAN_C0],
            c="r",
            marker="o",
            label="Bndry NDRs",
        )
        plt.legend(loc="upper right")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NDRS}/c0_scatter_chrm_{self._chrm.id}_ndr_{sr.min_ndr_len}"
            f"_bndrs_{sr.bndrs}.png"
        )

    def scatter_kmer(self, kmer: KMerSeq):
        PlotUtil.clearfig()
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_25)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        km = KMer(self._chrm)
        ax1.scatter(
            sr.non_bndry_ndrs()[LEN],
            km.count_w_rc(kmer, sr.non_bndry_ndrs()[START], sr.non_bndry_ndrs()[END])
            / sr.non_bndry_ndrs()[LEN],
            c="b",
            marker="s",
            label="Non-bndry NDRs",
        )
        ax1.scatter(
            sr.bndry_ndrs()[LEN],
            km.count_w_rc(kmer, sr.bndry_ndrs()[START], sr.bndry_ndrs()[END])
            / sr.bndry_ndrs()[LEN],
            c="r",
            marker="o",
            label="Bndry NDRs",
        )
        plt.legend(loc="upper right")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NDRS}/kmercnt_scatter_{kmer}_chrm_{self._chrm.id}_ndr_{sr.min_ndr_len}"
            f"_bndrs_{sr.bndrs}.png"
        )


class LineC0Plot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def line_c0_bndrs_indiv_toppings(self, btype: BoundariesType) -> None:
        bparm = BndParm.HIRS_WD if btype == BoundariesType.HEXP else BndFParm.WD_25
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(btype, bparm)
        for bndrs, pstr in zip(
            [sr.prmtr_bndrs(), sr.non_prmtr_bndrs()], ["prmtr", "nonprmtr"]
        ):
            for bndry in bndrs:
                self._line_c0_bndry_indiv_toppings(bndry, str(bndrs), pstr)

    def _line_c0_bndry_indiv_toppings(
        self, bndry: pd.Series, bstr: str, pstr: str
    ) -> Path:
        self.line_c0_toppings(getattr(bndry, START), getattr(bndry, END), save=False)
        plt.title(
            f"C0 around {pstr} boundary at {getattr(bndry, MIDDLE)} bp of chrm {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm.id}_{bstr}/"
            f"bndry_{pstr}_{getattr(bndry, START)}_{getattr(bndry, END)}_score_"
            f"{round(getattr(bndry, SCORE), 2)}.png"
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

    def helsep_box(self) -> Path:
        pass

    def dinc_explain_scatter(self) -> Path:
        sr = SubRegions(Chromosome("VL"))
        mc0 = sr.prmtrs[MEAN_C0]
        dinc = Dinc(sr.chrm)
        ta = self._total_dinc(sr.prmtrs, dinc.ta_count_multisegment)
        cg = self._total_dinc(sr.prmtrs, dinc.cg_count_multisegment)
        fig, axs = plt.subplots(2)
        fig.suptitle("Scatter plot of mean C0 vs TpA and CpG content in promoters")
        axs[0].scatter(mc0, ta)
        axs[1].scatter(mc0, cg)
        axs[0].set_ylabel("TpA content")
        axs[0].set_xlabel("Mean C0")
        axs[1].set_ylabel("CpG content")
        axs[1].set_xlabel("Mean C0")
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/prmtrs_ta_cg_scatter.png"
        )

    def dinc_explain_box(self) -> Path:
        subr = SubRegions(Chromosome("VL"))
        subr.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        dinc = Dinc(Chromosome("VL"))
        pmwb = subr.prmtrs_with_bndrs()
        pmob = subr.prmtrs_wo_bndrs()
        labels = ["Prmtrs w b", "Prmtrs wo b"]
        fig, axs = plt.subplots(3)
        fig.suptitle("TpA and CpG content in promoters")
        for axes in axs:
            axes.grid(which="both")
        PlotUtil.box_many(
            [pmwb[MEAN_C0], pmob[MEAN_C0]],
            labels=labels,
            ylabel="Mean C0",
            pltobj=axs[0],
        )
        for dinc, cnt_fnc, axes in zip(
            ["TpA", "CpG"],
            [dinc.ta_count_multisegment, dinc.cg_count_multisegment],
            axs[1:],
        ):
            PlotUtil.box_many(
                [
                    self._total_dinc(pmwb, cnt_fnc),
                    self._total_dinc(pmob, cnt_fnc),
                ],
                labels=labels,
                ylabel=f"{dinc} content",
                pltobj=axes,
            )

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/dinc_explain_VL.png"
        )

    def _total_dinc(self, rgns: Regions, cnt_fnc: Callable) -> NDArray[(Any,), float]:
        return np.array(cnt_fnc(rgns[START], rgns[END]))

    def _mean_dinc(self, rgns: Regions, cnt_fnc: Callable) -> NDArray[(Any,), float]:
        return self._total_dinc(rgns, cnt_fnc) / rgns[LEN].to_numpy()

    def both_sorted_motif_contrib(self):
        for i, num in enumerate(MotifsM30().sorted_contrib()):
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
