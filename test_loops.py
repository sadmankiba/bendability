from loops import Loops
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