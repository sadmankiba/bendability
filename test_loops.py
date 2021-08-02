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
        assert set(df.columns) == set(['start', 'end', 'res'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop._loop_file])
        assert len(df) == int(s.split()[0]) - 2


    def test_plot_mean_c0_across_loops(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_mean_c0_across_loops(150)
        path = Path('figures/chromosome/V_actual/loops/mean_c0_total_loop_perc_150_maxlen_100000.png')
        assert path.is_file()
    

    def test_plot_c0_in_individual_loop(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_c0_in_individual_loop()
        loop_df = loop._read_loops()
        resolutions = np.unique(loop_df['res'].to_numpy())
        for res in resolutions:
            assert Path(f'figures/chromosome/V_actual/loops/{res}').is_dir()


    def test_plot_c0_around_anchor(self):
        loop = Loops(Chromosome('VL'))
        loop.plot_c0_around_anchor(500)
        path = Path('figures/chromosome/V_actual/loops/mean_c0_loop_hires_anchor_dist_500_balanced.png')
        assert path.is_file()


    def test_plot_nuc(self):
        loop = Loops(Chromosome('II'))
        loop.plot_mean_nuc_occupancy_across_loops()
        path = Path('figures/chromosome/II_predicted/loops/mean_nucleosome_occupancy_total_loop_perc_150_maxlen_100000.png')
        assert path.is_file()


    def test_find_avg_c0(self):
        mean = Loops(Chromosome('VL')).find_avg_c0()
        print(mean)
        assert mean < 0
        assert mean > -0.3

class TestMultipleChrLoops:
    def test_multichr_find_avg_c0(self):
        MultiChrLoops().find_avg_c0()
        path = Path('data/generated_data/loops/multichr_c0_stat.tsv')
        assert path.is_file()
