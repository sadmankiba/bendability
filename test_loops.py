from loops import Loops, MultiChrLoops
from chromosome import Chromosome

import numpy as np

import unittest
import subprocess
from pathlib import Path

from prediction import Prediction

class TestLoops(unittest.TestCase):
    def test_read_loops(self):
        loop = Loops(Chromosome('VL', None))
        df = loop._read_loops()
        assert set(df.columns) == set(['start', 'end', 'res', 'len'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop._loop_file])
        assert len(df) == int(s.split()[0]) - 2


    def test_plot_mean_c0_across_loops(self):
        chr = Chromosome('VL', None)
        loop = Loops(chr)
        loop.plot_mean_c0_across_loops(150)
        path = Path(f'figures/loop/mean_c0_p_150_mxl_100000_{chr}.png')
        assert path.is_file()
    

    def test_plot_c0_in_individual_loop(self):
        loop = Loops(Chromosome('VL', None))
        loop.plot_c0_in_individual_loop()
        assert Path(f'figures/loop/VL').is_dir()


    def test_plot_c0_around_anchor(self):
        chr = Chromosome('VL', None)
        loop = Loops(chr)
        loop.plot_c0_around_anchor(500)
        path = Path(f'figures/loop_anchor/dist_500_{chr}.png')
        assert path.is_file()


    def test_plot_nuc_across_loops(self):
        chr = Chromosome('II', Prediction(30))
        loop = Loops(chr)
        loop.plot_mean_nuc_occupancy_across_loops()
        path = Path(f'figures/loop/mean_nuc_occ_p_150_mxl_100000_{chr}.png')
        assert path.is_file()


    def test_find_avg_c0(self):
        mean = Loops(Chromosome('VL', None)).find_avg_c0()
        print(mean)
        assert mean < 0
        assert mean > -0.3


    def test_find_avg_c0_in_quartile_by_pos(self):
        arr = Loops(Chromosome('VL', None)).find_avg_c0_in_quartile_by_pos()
        print(arr)
        assert arr.shape == (4,)


    def test_find_avg_around_anc(self):
        avg = Loops(Chromosome('VL', None)).find_avg_around_anc('start', 500)
        assert avg > -1
        assert avg < 1


class TestMultipleChrLoops:
    def test_multichr_find_avg_c0(self):
        MultiChrLoops().save_avg_c0_stat()
        path = Path('data/generated_data/loop/multichr_avg_c0_stat_30.tsv')
        assert path.is_file()
