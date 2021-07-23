from chrv import ChrV

import numpy as np 

import unittest 
from pathlib import Path

from constants import CHRVL_LEN, CHRV_TOTAL_BP

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


    def test_covering_sequences_at(self):
        chrv = ChrV()
        arr = chrv._covering_sequences_at(30)
        self.assertListEqual(arr.tolist(), [1, 2, 3, 4, 5])
        arr = chrv._covering_sequences_at(485)
        self.assertListEqual(arr.tolist(), [64, 65, 66, 67, 68, 69, 70])
        arr = chrv._covering_sequences_at(576860)
        self.assertListEqual(arr.tolist(), [82403, 82404])
    

    def test_spread_c0_balanced(self):
        chrv = ChrV()
        spread_c0 = chrv.spread_c0_balanced()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


    def test_spread_c0_weighted(self):
        chrv = ChrV()
        spread_c0 = chrv.spread_c0_weighted()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


if __name__ == '__main__':
    unittest.main()
