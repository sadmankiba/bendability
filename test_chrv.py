from chrv import ChrV

import numpy as np 

import unittest 

class TestChrV(unittest.TestCase):
    def test_moving_avg(self):
        chrv = ChrV()
        arr = np.array([4, 6, 1, -9, 2, 7, 3])
        ma = chrv._calc_moving_avg(arr, 4)
        self.assertListEqual(ma.tolist(), [0.5, 0, 0.25, 0.75])
        

if __name__ == '__main__':
    unittest.main()
