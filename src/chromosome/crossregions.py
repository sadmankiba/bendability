from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .chromosome import Chromosome
from .genes import Promoters
from .nucleosomes import Linkers
from conformation.domains import BoundariesHE
from util.util import PlotUtil, FileSave
from .regions import LEN, MEAN_C0

class CrossRegionsPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm

    def num_prmtrs_bndrs_ndrs(self) -> Path: 
        ndrs = Linkers(self._chrm).ndrs()
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.with_rgn(ndrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        bndrs = BoundariesHE(self._chrm)
        bndrs_with_ndr = bndrs.with_rgn(ndrs)
        bndrs_wo_ndr = bndrs - bndrs_with_ndr
        PlotUtil.clearfig()
        plt.bar([0,1,3,4], [len(prmtrs_with_ndr), len(prmtrs_wo_ndr), len(bndrs_with_ndr), len(bndrs_wo_ndr)])
        plt.xticks([0,1,3,4], ["Promoters with NDR", "Promoters without NDR", "Boundaries with NDR", "Boundaries without NDR"])
        plt.title(f"Promoters and Boundaries with and without NDR in {self._chrm.number}")
        return FileSave.figure_in_figdir(f"genes/num_prmtrs_bndrs_ndr_{self._chrm.number}.png")
        
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
            f"linkers/prob_distr_len_prmtrs_{self._chrm}.png"
        )
