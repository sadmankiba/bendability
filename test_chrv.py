from chrv import ChrV

import numpy as np 

import unittest 
from pathlib import Path

class TestChrV(unittest.TestCase):
    def test_moving_avg(self):
        chrv = ChrV()
        arr = np.array([4, 6, 1, -9, 2, 7, 3])
        ma = chrv._calc_moving_avg(arr, 4)
        self.assertListEqual(ma.tolist(), [0.5, 0, 0.25, 0.75])
    
    
    def test_plot_c0_vs_dist_from_dyad_spread(self):
        chrv = ChrV()
        chrv.plot_c0_vs_dist_from_dyad_spread(150)
        path = Path('figures/chrv/c0_dyad_dist_150_spread.png')
        self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()