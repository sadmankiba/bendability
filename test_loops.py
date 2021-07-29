from loops import Loops

import numpy as np

import unittest
import subprocess
from pathlib import Path

class TestLoops(unittest.TestCase):
    def test_read_loops(self):
        loop_file = 'juicer/data/generated_data/loops/merged_loops_r_500_1000_2000.bedpe'
        loops = Loops(loop_file)
        df = loops._read_loops()
        # self.assertCountEqual(df.columns.tolist(), ['start', 'end', 'res'])
        assert set(df.columns) == set(['start', 'end', 'res'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop_file])
        assert len(df) == int(s.split()[0]) - 2


    def test_plot_c0_vs_total_loop(self):
        loop = Loops()
        loop.plot_c0_vs_total_loop(150, 'actual')
        path = Path('figures/chrv/loops/actual_c0_total_loop_perc_150_maxlen_100000.png')
        assert path.is_file()
    
    
    def test_plot_chrv_c0_in_loops(self):
        loop = Loops()
        loop.plot_chrv_c0_in_loops()
        loop_df = loop._read_loops()
        resolutions = np.unique(loop_df['res'].to_numpy())
        for res in resolutions:
            assert Path(f'figures/chrv/loops/{res}').is_dir()


if __name__ == '__main__':
    unittest.main()