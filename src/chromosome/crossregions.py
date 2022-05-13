from __future__ import annotations
import random
import math
from pathlib import Path
from enum import Enum, auto
from collections import namedtuple
from typing import Any, Callable, Iterator, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.ticker as plticker
import pandas as pd
import numpy as np
from cairosvg import svg2png
from scipy.ndimage import gaussian_filter1d
from skimage.transform import resize
from nptyping import NDArray

from chromosome.chromosome import C0Spread, PlotChrm, ChrmOperator, Chromosome
from chromosome.genes import Genes, Promoters, STRAND
from chromosome.regions import END, MIDDLE, START, LEN, MEAN_C0, Regions
from chromosome.nucleosomes import Linkers, Nucleosomes
from models.prediction import Prediction
from motif.motifs import MotifsM30
from conformation.domains import (
    BndParmT,
    Boundaries,
    BoundariesF,
    BoundariesFN,
    BoundariesHE,
    SCORE,
    BndParm,
    BoundariesType,
    BoundariesFactory,
    BndFParm,
    BndSel,
    DomainsF,
    DomainsFN,
)
from conformation.loops import LoopAnchors, LoopInsides
from feature_model.helsep import DincUtil, HelSep, SEQ_COL
from util.util import Attr, PathObtain, PlotUtil, FileSave
from util.custom_types import PosOneIdx, KMerSeq
from util.kmer import KMer
from util.constants import (
    SEQ_LEN,
    FigSubDir,
    ONE_INDEX_START,
    GDataSubDir,
    YeastChrNumList,
)


class SubRegions:
    def __init__(self, chrm: Chromosome) -> None:
        self.chrm = chrm
        self._prmtrs = None
        self._bndrs = None
        self.bsel = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        self.min_ndr_len = 40

    @property
    def bndrs(self) -> Boundaries:
        def _bndrs():
            return BoundariesFactory(self.chrm).get_bndrs(self.bsel)

        return Attr.calc_attr(self, "_bndrs", _bndrs)

    @property
    def dmns(self) -> Promoters:
        def _dmns():
            if isinstance(self.bndrs, BoundariesFN):
                D = DomainsFN
            elif isinstance(self.bndrs, BoundariesF):
                D = DomainsF

            return D(self.bndrs)

        return Attr.calc_attr(self, "_dmns", _dmns)

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

    def lnks_in_bnds(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.bndrs)

    def lnks_in_dmns(self) -> Linkers:
        return self.lnkrs.mid_contained_in(self.dmns)

    def nucs_in_bnds(self) -> Nucleosomes:
        return self.nucs.mid_contained_in(self.bndrs)

    def nucs_in_dmns(self) -> Nucleosomes:
        return self.nucs.mid_contained_in(self.dmns)

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


class LabeledMC0Distribs:
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
                return self._sr.nucs[MEAN_C0], "Nucleosomes"
            if d == Distrib.NUCS_B:
                return self._sr.bndry_nucs()[MEAN_C0], "Nucleosomes\nin boundaries"
            if d == Distrib.NUCS_NB:
                return self._sr.non_bndry_nucs()[MEAN_C0], "Nucleosomes\nin domains"
            if d == Distrib.LNKRS:
                return self._sr.lnkrs[MEAN_C0], "lnkrs"
            if d == Distrib.NDRS:
                return self._sr.ndrs[MEAN_C0], "NDRs"
            if d == Distrib.NDRS_B:
                return self._sr.bndry_ndrs()[MEAN_C0], "NDRs in boundaries"
            if d == Distrib.NDRS_NB:
                return self._sr.non_bndry_ndrs()[MEAN_C0], "NDRs in domains"
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
        self._sr = SubRegions(chrm)

    def box_mean_c0_bndrs(self) -> Path:
        typ = "box"
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        sr.bsel = bsel_fanc
        ld = LabeledMC0Distribs(sr)
        grp_bndrs_nucs = {
            "dls": ld.dl(
                [
                    # Distrib.BNDRS,
                    # Distrib.BNDRS_E_100,
                    # Distrib.BNDRS_E_N50,
                    # Distrib.NUCS,
                    # Distrib.NUCS_B,
                    # Distrib.NUCS_NB,
                    # Distrib.LNKRS,
                    Distrib.NDRS,
                    Distrib.NDRS_B,
                    Distrib.NDRS_NB,
                ]
            ),
            "title": "",  # "Mean C0 distribution of boundaries, nucleosomes and NDRS",
            "fname": f"bndrs_nucs_{sr.bndrs}_{typ}_chrm_{sr.chrm}.png",
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
        return self.box_mean_c0(grp_bndrs_nucs, typ)

    @classmethod
    def box_mean_c0(cls, grp: dict, typ: Literal["box"] | Literal["violin"] = "box"):
        limy = False
        PlotUtil.font_size(14)
        PlotUtil.show_grid()
        distribs = [d for d, _ in grp["dls"]]
        labels = [l for _, l in grp["dls"]]

        fig, ax = plt.subplots()
        if typ == "box":
            plt.boxplot(distribs, showfliers=True, widths=0.5)
        elif typ == "violin":
            ax.violinplot(distribs, showmedians=True, showextrema=True)

        if limy:
            plt.ylim(-0.5, 0.1)
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, wrap=True)
        plt.ylabel("Mean C0")
        plt.title(grp["title"])
        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/c0_box/{grp['fname']}", sizew=7, sizeh=6
        )

    def box_bnd_dmn_lnklen(self):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANCN, BndFParm.SHR_50_LNK_0)
        dmns = DomainsFN(sr.bndrs)
        bl = sr.lnkrs.mid_contained_in(sr.bndrs)
        dl = sr.lnkrs.mid_contained_in(dmns)
        assert len(bl) + len(dl) == len(sr.lnkrs)

        plt.boxplot([dl[LEN], bl[LEN]], showfliers=True, widths=0.5)
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/lnklen_box_bnd_dmn_{sr.bndrs}.png"
        )

    def prob_distrib_mean_c0_bndrs(self):
        sr = SubRegions(self._chrm)
        bsel_hexp = BndSel(BoundariesType.HEXP, BndParm.HIRS_SHR)
        bsel_fanc = BndSel(BoundariesType.FANC, BndFParm.SHR_25)
        sr.bsel = bsel_hexp
        ld = LabeledMC0Distribs(sr)
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

    def prob_distrib_bndrs_nearest_ndr_distnc(self, min_lnker_len: int) -> Path:
        allchrm = True
        llen = min_lnker_len
        if allchrm:
            dists = np.empty((0,))
            pred = Prediction(35)
            for c in YeastChrNumList:
                print(c)
                chrm = Chromosome(c, prediction=pred, spread_str=C0Spread.mcvr)
                self._sr = SubRegions(chrm)
                self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
                dists = np.append(
                    dists,
                    self._sr.bndrs.nearest_locs_distnc(
                        self._sr.lnkrs.ndrs(llen)[MIDDLE]
                    ),
                )
        else:
            self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
            dists = self._sr.bndrs.nearest_locs_distnc(
                self._sr.lnkrs.ndrs(llen)[MIDDLE]
            )

        in_bnd = round(
            np.sum((-self._sr.bndrs.lim < dists) & (dists <= self._sr.bndrs.lim))
            / len(dists)
            * 100,
            1,
        )
        lft_bnd = round(np.sum(dists <= -self._sr.bndrs.lim) / len(dists) * 100, 1)
        rgt_bnd = round(np.sum(self._sr.bndrs.lim < dists) / len(dists) * 100, 1)
        PlotUtil.font_size(20)
        PlotUtil.prob_distrib(dists, label=str(llen))
        PlotUtil.vertline(-self._sr.bndrs.lim + 1, "k", linewidth=2)
        PlotUtil.vertline(self._sr.bndrs.lim, "k", linewidth=2)
        ax = plt.gca()
        ax.set_xlim([-250, 250])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        plt.text(0, sum(ylim) / 2, f"{in_bnd}%")
        plt.text((xlim[0] + -self._sr.bndrs.lim) / 2, sum(ylim) / 2, f"{lft_bnd}%")
        plt.text((xlim[1] + self._sr.bndrs.lim) / 2, sum(ylim) / 2, f"{rgt_bnd}%")

        # PlotUtil.show_grid()
        plt.xlabel("Distance from boundary middle (bp)")
        plt.ylabel("Density")
        # plt.title(
        #     f"Prob distrib of distance from boundary res={self._sr.bndrs.res} bp "
        #     f"middle to nearest NDR >= x bp"
        # )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/distnc_ndr_prob_distrib_{self._sr.bndrs}_"
            f"{f'all{str(self._chrm)[:-4]}' if allchrm else self._chrm}.png"
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


class DistribC0DistPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def bndrs(self, pltlim=500, dist=200):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)

        starts = np.array(range(-pltlim, pltlim, dist))
        box_starts = np.array(sr.bndrs[MIDDLE])[:, np.newaxis] + starts[np.newaxis, :]
        assert box_starts.shape == (len(sr.bndrs), math.ceil(2 * pltlim / dist))
        box_ends = box_starts + dist - 1
        pos_boxs = starts + int(dist / 2)
        c0s = []
        for s_boxs, e_boxs in zip(box_starts.T, box_ends.T):
            c0s.append(ChrmOperator(sr.chrm).c0_rgns(s_boxs, e_boxs).flatten())

        plt.boxplot(c0s, showfliers=False)
        plt.ylim(-0.5, 0.1)
        # plt.xticks(
        #     ticks=pos_boxs,
        # )
        plt.ylabel("C0 distrib")
        plt.title(
            f"C0 box distrib at distances among boundaries in chromosome {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm.id}_{str(sr.bndrs)}/"
            f"c0_box_distrib_pltlim_{pltlim}_dist_{dist}.png"
        )


class ScatterPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def scatter_c0(self) -> Path:
        PlotUtil.clearfig()
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)

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

        ax1.scatter(
            sr.non_bndry_ndrs()[LEN],
            KMer.count_w_rc(
                kmer, sr.chrm.seqf(sr.non_bndry_ndrs()[START], sr.non_bndry_ndrs()[END])
            )
            / sr.non_bndry_ndrs()[LEN],
            c="b",
            marker="s",
            label="Non-bndry NDRs",
        )
        ax1.scatter(
            sr.bndry_ndrs()[LEN],
            KMer.count_w_rc(
                kmer, sr.chrm.seqf(sr.bndry_ndrs()[START], sr.bndry_ndrs()[END])
            )
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
        self._sr = SubRegions(self._chrm)

    def line_c0_mean_lnks(self, pltlim=100) -> Path:
        self._sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        dmns = DomainsFN(self._sr.bndrs)
        bl = self._sr.lnkrs.mid_contained_in(self._sr.bndrs)
        dl = self._sr.lnkrs.mid_contained_in(dmns)
        assert len(bl) + len(dl) == len(self._sr.lnkrs)

        cop = ChrmOperator(self._chrm)
        # c0m = self._chrm.mean_c0_around_bps(
        #     cop.in_lim(self._sr.lnkrs[MIDDLE], pltlim), pltlim, pltlim
        # )
        c0m_b = self._chrm.mean_c0_around_bps(
            cop.in_lim(bl[MIDDLE], pltlim), pltlim, pltlim
        )
        c0m_d = self._chrm.mean_c0_around_bps(
            cop.in_lim(dl[MIDDLE], pltlim), pltlim, pltlim
        )
        PlotUtil.clearfig()
        PlotUtil.font_size(20)
        x = np.arange(2 * pltlim + 1) - pltlim
        # PlotChrm(self._chrm).plot_avg()

        # plt.plot(x, c0m, label="lnk all")
        plt.plot(x, c0m_b, label="linkers at boundaries", color="tab:blue", linewidth=2)
        plt.plot(x, c0m_d, label="linkers in domains", color="tab:red", linewidth=2)
        plt.legend()
        PlotUtil.vertline(0, "tab:orange")
        # PlotUtil.show_grid()
        plt.xlabel("Distance from linker middle (bp)")
        plt.ylabel("Cyclizability (C0)")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.LINKERS}/{self._chrm}_{str(self._sr.lnkrs)}/"
            f"c0_line_mean_pltlim_{pltlim}.png"
        )

    def line_c0_mean_bndrs(self, pltlim=100) -> Path:
        show_legend = False
        smooth = False
        self._sr.bsel = BndSel(BoundariesType.FANCN, BndFParm.SHR_50_LNK_0)
        C0MeanArr = namedtuple("C0MeanArr", ["val", "label"])
        c0m_bndrs = C0MeanArr(
            self._chrm.mean_c0_around_bps(self._sr.bndrs[MIDDLE], pltlim, pltlim), "all"
        )
        c0m_bndrs_p = C0MeanArr(
            self._chrm.mean_c0_around_bps(
                self._sr.prmtr_bndrs()[MIDDLE], pltlim, pltlim
            ),
            "bndrs_p",
        )
        c0m_bndrs_np = C0MeanArr(
            self._chrm.mean_c0_around_bps(
                self._sr.non_prmtr_bndrs()[MIDDLE], pltlim, pltlim
            ),
            "bndrs_np",
        )

        c0ms = [c0m_bndrs]

        PlotUtil.clearfig()
        x = np.arange(2 * pltlim + 1) - pltlim

        if smooth:
            plt.plot(x, c0m_bndrs.val, color="tab:gray", alpha=0.5)
            plt.plot(x, gaussian_filter1d(c0m_bndrs.val, 40), color="black")
        else:
            for cm in c0ms:
                plt.plot(x, cm.val, label=cm.label)

        PlotChrm(self._chrm).plot_avg()
        if show_legend:
            plt.legend()
        PlotUtil.show_grid()
        plt.xlabel("Distance from boundary middle(bp)")
        plt.ylabel("C0")
        plt.title(f"C0 mean around boundaries in chromosome {self._chrm.id}")

        FileSave.nptxt(
            c0m_bndrs.val,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.BOUNDARIES}/"
            f"{self._chrm}_{self._sr.bndrs}/chrm{self._chrm.id}_c0_line_mean_pltlim_{pltlim}.txt",
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{str(self._sr.bndrs)}/"
            f"c0_line_mean_pltlim_{pltlim}.png"
        )

    def line_c0_bndrs_indiv_toppings(self) -> None:
        self._sr.bsel = BndSel(BoundariesType.FANCN, BndFParm.SHR_50)
        for bndrs, pstr in zip(
            [self._sr.prmtr_bndrs(), self._sr.non_prmtr_bndrs()], ["prmtr", "nonprmtr"]
        ):
            for bndry in bndrs:
                self._line_c0_bndry_indiv_toppings(bndry, str(bndrs), pstr)

    def _line_c0_bndry_indiv_toppings(
        self, bndry: pd.Series, bstr: str, pstr: str
    ) -> Path:
        self.line_c0_toppings(
            getattr(bndry, START) - 100, getattr(bndry, END) + 100, save=False
        )
        plt.title(
            f"C0 around {pstr} boundary at {getattr(bndry, MIDDLE)} bp of chrm {self._chrm.id}"
        )
        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{self._chrm}_{bstr}/"
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

    def line_c0_lpancs_indiv_toppings(self) -> None:
        for lpanc in self._sr.lpancrs:
            self._line_c0_lpancs_indiv_toppings(lpanc, str(self._sr.lpancrs))

    def _line_c0_lpancs_indiv_toppings(self, lpanc: pd.Series, lpstr: str) -> Path:
        self.line_c0_toppings(getattr(lpanc, START), getattr(lpanc, END), save=False)
        return FileSave.figure_in_figdir(
            f"{FigSubDir.LOOP_ANCHORS}/{self._chrm}_{lpstr}/"
            f"{getattr(lpanc, START)}_{getattr(lpanc, END)}.png"
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
            wb = 50
            hb = 0.1
            for m, s in zip(mids, scr):
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
        dyads = self._sr.nucs[MIDDLE]

        colors = ["tab:orange", "tab:brown", "tab:purple", "tab:green"]
        labels = ["nuc", "bndrs", "tss", "lng lnk"]
        _nuc_ellipse(_within(dyads), colors[0])
        _bndrs(_within(self._sr.bndrs[MIDDLE]), self._sr.bndrs[SCORE], colors[1])
        _tss(_within(self._sr.genes.frwrd_genes()[START]), True, colors[2])
        _tss(_within(self._sr.genes.rvrs_genes()[END]), False, colors[2])
        _lng_linkrs(_end_within(self._sr.ndrs), colors[3])
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


class PaperLineC0Plot:
    @classmethod
    def bnd(cls):
        chrm = Chromosome("VL", spread_str=C0Spread.mcvr)
        cop = ChrmOperator(chrm)
        sr = SubRegions(chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        # 341 - 481 linker, 401 - 600 boundary
        # s, e =  303341, 303600
        # ls, le = 303341, 303481
        # bs, be = 303401, 303600
        # lg = -0.4
        # bg = -0.5
        s, e = 59201, 59476
        ls, le = 59339, 59476
        bs, be = 59201, 59400
        lg = -0.2
        bg = -0.3
        c0 = cop.c0(s, e)
        x = np.arange(s, e + 1)
        PlotUtil.font_size(20)
        plt.plot(x, c0)
        plt.xlabel("Position in bp")
        plt.ylabel("Cyclizability")

        line = plt.Polygon(
            [[ls, lg], [le, lg]], closed=None, fill=None, edgecolor="tab:blue", lw=3
        )
        plt.gca().add_patch(line)
        plt.text(
            (ls + le) / 2,
            lg,
            "linker",
            color="tab:blue",
            ha="center",
            va="top",
        )

        line = plt.Polygon(
            [[bs, bg], [be, bg]], closed=None, fill=None, edgecolor="tab:red", lw=3
        )
        plt.gca().add_patch(line)
        plt.text(
            (bs + be) / 2,
            bg,
            "boundary",
            color="tab:red",
            ha="center",
            va="top",
        )

        FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/{chrm}_{sr.bndrs}/bndrs_{s}_{e}.png"
        )


class MCLineC0Plot:
    pred = Prediction(35)

    @classmethod
    def line_c0_mean_lnks_nucs(cls, pltlim=100):
        nucs = True
        mcb_c0 = []
        mcd_c0 = []
        for c in YeastChrNumList:
            print(c)
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
            cop = ChrmOperator(chrm)

            b, d = (
                (sr.nucs_in_bnds(), sr.nucs_in_dmns())
                if nucs
                else (sr.lnks_in_bnds(), sr.lnks_in_dmns())
            )

            mcb_c0.append(
                cop.c0_rgns(
                    cop.in_lim(b[MIDDLE], pltlim) - pltlim,
                    cop.in_lim(b[MIDDLE], pltlim) + pltlim,
                )
            )
            mcd_c0.append(
                cop.c0_rgns(
                    cop.in_lim(d[MIDDLE], pltlim) - pltlim,
                    cop.in_lim(d[MIDDLE], pltlim) + pltlim,
                )
            )

        mc0b = np.vstack(mcb_c0).mean(axis=0)
        mc0d = np.vstack(mcd_c0).mean(axis=0)

        x = np.arange(2 * pltlim + 1) - pltlim
        PlotUtil.font_size(20)
        labels = (
            ["Nucleosomes at boundaries", "Nucleosomes in domains"]
            if nucs
            else ["Linkers at boundaries", "Linkers in domains"]
        )
        plt.plot(x, mc0b, color="tab:blue", label=labels[0], linewidth=2)
        plt.plot(x, mc0d, color="tab:red", label=labels[1], linewidth=2)
        PlotUtil.vertline(0, "tab:orange", linewidth=2)
        plt.legend()
        plt.xlabel(f"Distance from {'nucleosome' if nucs else 'linker'} middle (bp)")
        plt.ylabel("Cyclizability (C0)")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.NUCLEOSOMES if nucs else FigSubDir.LINKERS}/c0_line_mean_all{str(sr.chrm)[:-4]}_{sr.bndrs}_pltlim_{pltlim}.png"
        )

    @classmethod
    def _sr(cls, c: str):
        chrm = Chromosome(c, prediction=cls.pred, spread_str=C0Spread.mcvr)
        return SubRegions(chrm), chrm

    @classmethod
    def line_c0_mean_bndrs(cls, pltlim=100):
        mcbndrs_c0 = []
        for c in YeastChrNumList:
            sr, chrm = cls._sr(c)
            sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
            # mcbndrs_c0 += sr.bndrs.c0()
            mcbndrs_c0.append(
                ChrmOperator(chrm).c0_rgns(
                    (sr.bndrs[MIDDLE] - pltlim).tolist(),
                    (sr.bndrs[MIDDLE] + pltlim).tolist(),
                )
            )

        mc0 = np.vstack(mcbndrs_c0).mean(axis=0)

        FileSave.nptxt(
            mc0,
            f"{PathObtain.gen_data_dir()}/{GDataSubDir.BOUNDARIES}/"
            f"c0_line_mean_all{str(sr.chrm)[:-4]}_{sr.bndrs}_pltlim_{pltlim}.txt",
        )

        x = np.arange(2 * pltlim + 1) - pltlim
        plt.plot(x, mc0)
        PlotUtil.show_grid()
        plt.xlabel("bp in boundaries")
        plt.ylabel("C0")
        plt.title(f"C0 mean around boundaries")

        return FileSave.figure_in_figdir(
            f"{FigSubDir.BOUNDARIES}/c0_line_mean_all{str(sr.chrm)[:-4]}_{sr.bndrs}_pltlim_{pltlim}.png"
        )


class SegmentLineC0Plot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def sl_lnkrs(self):
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        lnks = [sr.lnkrs.len_in(mn=st, mx=st + 20 - 1) for st in (11, 31, 51, 71)]
        lnksb = [lnk.mid_contained_in(sr.bndrs) for lnk in lnks]
        lnksd = [lnk - lnkb for lnk, lnkb in zip(lnks, lnksb)]
        lnksbc0 = [
            np.array([resize(c0_arr, (101,)) for c0_arr in lnkb.c0()]).mean(axis=0)
            for lnkb in lnksb
        ]
        lnksdc0 = [
            np.array([resize(c0_arr, (101,)) for c0_arr in lnkd.c0()]).mean(axis=0)
            for lnkd in lnksd
        ]
        fig, ax = plt.subplots(4, 2)
        for i in range(4):
            ax[i][0].plot(lnksbc0[i])
            ax[i][1].plot(lnksdc0[i])

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/linkers/lnkrs_c0_segm_bndrs_dmns_{self._chrm}.png"
        )


RIGID_PAIRS = [
    ("GC", "TT"),
    ("AA", "GC"),
    ("GC", "TA"),
    ("AT", "GC"),
    ("AA", "CG"),
    ("AA", "CC"),
    ("GG", "TT"),
    ("CG", "TT"),
    ("TT", "CC"),
    ("TA", "CC"),
]

FLEXIBLE_PAIRS = [
    ("CC", "CC"),
    ("GC", "CG"),
    ("GC", "CC"),
    ("TA", "TT"),
    ("AT", "TT"),
    ("AA", "TA"),
    ("AA", "AA"),
    ("AA", "TT"),
    ("TT", "TT"),
    ("GC", "GC"),
]


class LinePlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def helsep_mean_bndrs(self, pltlim=100):
        rigid = False

        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        hs = HelSep()
        harr = np.empty((len(sr.bndrs), 2 * pltlim + 1))
        for i, b in enumerate(sr.bndrs):
            dfh = hs.helical_sep_of(
                list(
                    map(
                        lambda p: sr.chrm.seqf(
                            p - SEQ_LEN / 2,
                            p + SEQ_LEN / 2 - 1,
                        ),
                        range(
                            getattr(b, MIDDLE) - pltlim, getattr(b, MIDDLE) + pltlim + 1
                        ),
                    )
                )
            )
            pairs = RIGID_PAIRS if rigid else FLEXIBLE_PAIRS
            cols = [DincUtil.pair_str(*p) for p in pairs]
            harr[i] = np.sum(dfh[cols].to_numpy(), axis=1)

        harr = harr.mean(axis=0)
        x = np.arange(2 * pltlim + 1) - pltlim
        pt = "rigid" if rigid else "flexible"
        PlotUtil.show_grid()
        plt.plot(x, harr)
        plt.xlabel("Position from boundary mid")
        plt.ylabel(f"Helsep {pt} content")
        plt.title(f"Helsep {pt} content in bndrs")
        return FileSave.figure_in_figdir(
            f"{sr.bndrs.fig_subdir()}/helsep_{pt}_content_pltlim_{pltlim}.png"
        )

    def dinc_mean_bndrs(self, pltlim=100):
        rc = False
        sr = SubRegions(self._chrm)
        sr.bsel = BndSel(BoundariesType.FANC, BndFParm.SHR_50)
        seqs = sr.chrm.seqf(sr.bndrs[MIDDLE] - pltlim, sr.bndrs[MIDDLE] + pltlim)
        arr = np.zeros((len(seqs), len(seqs[0])))
        dinc = "CG"
        pos_func = KMer.find_pos_w_rc if rc else KMer.find_pos
        for i, seq in enumerate(seqs):
            arr[i, pos_func(dinc, seq)] = 1

        arr = arr.mean(axis=0)
        x = np.arange(2 * pltlim + 1) - pltlim
        plt.plot(x, arr)
        plt.xlabel("Position from boundary mid")
        plt.ylabel("Content")
        plt.title(f"{dinc} content in bndrs")
        return FileSave.figure_in_figdir(
            f"{sr.bndrs.fig_subdir()}/{dinc}_content_pltlim_{pltlim}_rc_{rc}.png"
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

        ta = sum(KMer.count("TA", sr.chrm.seqf(sr.prmtrs[START], sr.prmtrs[END])))
        cg = sum(KMer.count("CG", sr.chrm.seqf(sr.prmtrs[START], sr.prmtrs[END])))
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
        for dinc, axes in zip(
            ["TA", "CG"],
            axs[1:],
        ):
            PlotUtil.box_many(
                [
                    sum(KMer.count(dinc, subr.chrm.seqf(pmwb[START], pmwb[END]))),
                    sum(KMer.count(dinc, subr.chrm.seqf(pmob[START], pmob[END]))),
                ],
                labels=labels,
                ylabel=f"{dinc} content",
                pltobj=axes,
            )

        return FileSave.figure_in_figdir(
            f"{FigSubDir.CROSSREGIONS}/dinc_explain_VL.png"
        )

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
