from constants import CHRV_TOTAL_BP
from loops import Loops, PlotLoops
from chromosome import Chromosome

import unittest
import subprocess
from pathlib import Path

from prediction import Prediction


class TestLoops(unittest.TestCase):
    def test_len(self):
        loops = Loops(Chromosome('VL', None))
        assert len(loops) == len(loops._loop_df)
    
    def test_getitem(self): 
        loops = Loops(Chromosome('VL', None))
        assert loops[10].tolist() == loops._loop_df.iloc[10].tolist()

    def test_read_loops(self):
        loop = Loops(Chromosome('VL', None))
        df = loop._read_loops()
        assert set(df.columns) == set(['start', 'end', 'res', 'len'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop._loop_file])
        assert len(df) == int(s.split()[0]) - 2

    def test_add_mean_c0_val(self):
        """Assert mean C0 is: linker < full < nuc"""
        loops = Loops(Chromosome('VL'))
        mean_cols = loops.add_mean_c0()
        assert all(list(map(lambda col: col in loops._loop_df.columns, mean_cols)))
        assert loops._loop_df[mean_cols[2]].mean() < loops._loop_df[mean_cols[0]].mean()
        assert loops._loop_df[mean_cols[1]].mean() > loops._loop_df[mean_cols[0]].mean()

    def test_add_mean_c0_type_conserve(self):
        loops = Loops(Chromosome('VL'))
        loops.add_mean_c0()
        dtypes = loops._loop_df.dtypes
        assert dtypes['start'] == int 
        assert dtypes['end'] == int 
        assert dtypes['len'] == int 

    def test_exclude_above_len(self):
        bf_loops = Loops(Chromosome('VL', None))
        bf_len = len(bf_loops._loop_df)
        bf_arr = bf_loops.get_loop_cover(bf_loops._loop_df)

        af_loops = Loops(Chromosome('VL', None), 100000)
        af_len = len(af_loops._loop_df)
        af_arr = af_loops.get_loop_cover(af_loops._loop_df)
        
        assert bf_len >= af_len
        assert bf_arr.sum() >= af_arr.sum()

    def test_get_loop_cover(self):
        loops = Loops(Chromosome('VL', None))
        chrm_arr = loops.get_loop_cover(loops._loop_df)
        assert chrm_arr.shape == (CHRV_TOTAL_BP, )
        perc = chrm_arr.sum() / CHRV_TOTAL_BP * 100
        assert 10 < perc < 90

    def test_plot_scatter_mean_c0_nuc_linker_individual_loop(self):
        path = Loops(Chromosome('VL')).plot_scatter_mean_c0_nuc_linker_individual_loop()
        assert path.is_file()


class TestPlotLoops:
    def test_plot_c0_in_individual_loop(self):
        ploops = PlotLoops(Chromosome('VL'))
        paths = ploops.plot_c0_in_individual_loop()
        for path in paths: 
            assert path.is_file()
    
    def test_plot_mean_c0_across_loops(self):
        chr = Chromosome('VL', None)
        ploops = PlotLoops(chr)
        path = ploops.plot_mean_c0_across_loops(150)
        assert path.is_file()

    def test_plot_nuc_occ_across_loops(self):
        chr = Chromosome('II', Prediction(30))
        ploops = PlotLoops(chr)
        path = ploops.plot_mean_nuc_occupancy_across_loops()
        assert path.is_file()
    
    def test_plot_c0_around_anchor(self):
        ploops = PlotLoops(Chromosome('VL'))
        path = ploops.plot_c0_around_anchor(500)
        assert path.is_file()

    def test_plot_c0_around_individual_anchors(self):
        ploops = PlotLoops(Chromosome('VL'))
        paths = ploops.plot_c0_around_individual_anchors()
        for path in paths: 
            assert path.is_file()
    

