from loops import Loops, MultiChrLoops
from chromosome import Chromosome

import numpy as np

import unittest
import subprocess
from pathlib import Path

class TestLoops(unittest.TestCase):
    def test_read_loops(self):
        loop = Loops(Chromosome('VL'))
        df = loop._read_loops()
        assert set(df.columns) == set(['start', 'end', 'res', 'len'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop._loop_file])
        assert len(df) == int(s.split()[0]) - 2


    def test_plot_mean_c0_across_loops(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_mean_c0_across_loops(150)
        path = Path('figures/loop/mean_c0_p_150_mxl_100000_VL.png')
        assert path.is_file()
    

    def test_plot_c0_in_individual_loop(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_c0_in_individual_loop()
        assert Path(f'figures/loop/VL').is_dir()


    def test_plot_c0_around_anchor(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_c0_around_anchor(500)
        path = Path('figures/loop_anchor/dist_500_VL.png')
        assert path.is_file()


    def test_plot_nuc_across_loops(self):
        loop = Loops(Chromosome('II'))
        loop.plot_mean_nuc_occupancy_across_loops()
        path = Path('figures/loop/mean_nuc_occ_p_150_mxl_100000_II.png')
        assert path.is_file()


    def test_find_avg_c0(self):
        mean = Loops(Chromosome('VL')).find_avg_c0()
        print(mean)
        assert mean < 0
        assert mean > -0.3


    def test_find_avg_c0_in_quartile_by_pos(self):
        arr = Loops(Chromosome('VL')).find_avg_c0_in_quartile_by_pos()
        print(arr)
        assert arr.shape == (4,)


class TestMultipleChrLoops:
    def test_multichr_find_avg_c0(self):
        MultiChrLoops().find_avg_c0()
        path = Path('data/generated_data/loop/multichr_c0_stat.tsv')
        assert path.is_file()
