from pathlib import Path 

import numpy as np
import pandas as pd 
import pytest

from conformation.coverloops import CoverLoops, MCCoverLoops, NonCoverLoops, PlotCoverLoops, MultiChrmLoopsCoverCollector, PlotMCCoverLoops
from conformation.loops import COL_MEAN_C0_FULL, Loops, COL_START, COL_END, MCLoops
from chromosome.chromosome import Chromosome, MultiChrm
from util.constants import ONE_INDEX_START

@pytest.fixture
def cloops_vl(loops_vl):
    return CoverLoops(loops_vl)

class TestCoverLoops:
    def test_loops_from_cover(self, cloops_vl):
        lcv = np.array([False, False, True, False, True, True, False, True, False])
        df = cloops_vl._loops_from_cover(lcv)
        assert df[COL_START].tolist() == [3, 5, 8]
        assert df[COL_END].tolist() == [3, 6, 8]

    def test_iter(self, cloops_vl):
        cloops = [cl for cl in cloops_vl]
        assert len(cloops) > 0
        assert getattr(cloops[0], COL_START) > ONE_INDEX_START
    
    def test_access(self, cloops_vl):
        assert len(cloops_vl) == len(cloops_vl[COL_MEAN_C0_FULL])

class TestNonCoverLoops:
    def test_noncoverloops(self, loops_vl, cloops_vl):
        ncloops = NonCoverLoops(loops_vl)
        assert len(ncloops) == len(cloops_vl) + 1

class TestPlotCoverLoops:
    def test_plot_histogram_c0(self):
        ploops = PlotCoverLoops(Chromosome("VL"))
        figpath = ploops.plot_histogram_c0()
        assert figpath.is_file()
    
class TestMCCoverLoops:
    def test_creation(self, cloops_vl):
        mccl = MCCoverLoops(MCLoops(MultiChrm(("VL", "I"))))
        cli = CoverLoops(Loops(Chromosome('I')))
        assert len(mccl) == len(cloops_vl) + len(cli)

class TestPlotMCCoverLoops:
    def test_histogram(self):
        figpath = PlotMCCoverLoops(MultiChrm(('VL', 'I'))).plot_histogram_c0()
        assert figpath.is_file()

class TestMultiChrmLoopsCoverCollector:
    def test_get_cover_stat(self):
        colt_df, path_str = MultiChrmLoopsCoverCollector(
            ("VL",), 1000000
        ).get_cover_stat()
        assert Path(path_str).is_file()
        assert pd.read_csv(path_str, sep="\t").columns.tolist() == [
            "ChrID",
            "loop_nuc",
            "loop_linker",
            "non_loop_nuc",
            "non_loop_linker",
        ]

    def test_plot_cover_stat(self):
        path_str = MultiChrmLoopsCoverCollector(("VL",), 100000).plot_bar_cover_stat()
        assert Path(path_str).is_file()
