from pathlib import Path 

import numpy as np
import pandas as pd 
import pytest

from conformation.coverloops import CoverLoops, PlotCoverLoops, MultiChrmLoopsCoverCollector
from conformation.loops import Loops, COL_START, COL_END
from chromosome.chromosome import Chromosome
from util.constants import ONE_INDEX_START


class TestCoverLoops:
    @pytest.fixture
    def cloops_vl(self):
        return CoverLoops(Loops(Chromosome("VL")))

    def test_noncoverloops(self, cloops_vl):
        assert len(cloops_vl.noncoverloops_with_c0()) == len(cloops_vl._cloops) + 1

    def test_loops_from_cover(self, cloops_vl):
        lcv = np.array([False, False, True, False, True, True, False, True, False])
        df = cloops_vl._loops_from_cover(lcv)
        assert df[COL_START].tolist() == [3, 5, 8]
        assert df[COL_END].tolist() == [3, 6, 8]

    def test_iter(self, cloops_vl):
        cloops = [cl for cl in cloops_vl]
        assert len(cloops) > 0
        assert getattr(cloops[0], COL_START) > ONE_INDEX_START


class PlotCoverLoopsTest:
    def test_plot_histogram_c0(self):
        ploops = PlotCoverLoops(Chromosome("VL"))
        figpath = ploops.plot_histogram_c0()
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
