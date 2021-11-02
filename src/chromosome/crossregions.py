from pathlib import Path 

import matplotlib.pyplot as plt 

from .chromosome import Chromosome
from .genes import Promoters
from .nucleosomes import Linkers
from util.util import PlotUtil, FileSave
from .regions import LEN 


class CrossRegionsPlot:
    def __init__(self, chrm: Chromosome) -> None:
        self._chrm = chrm
        
    def prob_distrib_prmtr_long_linkers(self) -> Path: 
        lnkrs = Linkers(self._chrm)
        lng_lnkrs = lnkrs.len_at_least(80)
        prmtrs = Promoters(self._chrm)
        prmtrs_with_ndr = prmtrs.rgns_contain_any_rgn(lng_lnkrs)
        prmtrs_wo_ndr = prmtrs - prmtrs_with_ndr
        pass

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
