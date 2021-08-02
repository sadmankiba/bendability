from chromosome import Chromosome, ChromosomeUtil, Spread

import numpy as np 

import unittest 
from pathlib import Path

from constants import CHRVL_LEN, CHRV_TOTAL_BP


class TestChromosomeUtil(unittest.TestCase):
    def test_moving_avg(self):
        chr_util = ChromosomeUtil()
        arr = np.array([4, 6, 1, -9, 2, 7, 3])
        ma = chr_util.calc_moving_avg(arr, 4)
        self.assertListEqual(ma.tolist(), [0.5, 0, 0.25, 0.75])
    
    def test_get_total_bp(self):
        assert ChromosomeUtil().get_total_bp(5) == 78


class TestSpread(unittest.TestCase):
    def test_covering_sequences_at(self):
        chrv = Chromosome('VL')
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        arr = spread._covering_sequences_at(30)
        self.assertListEqual(arr.tolist(), [1, 2, 3, 4, 5])
        arr = spread._covering_sequences_at(485)
        self.assertListEqual(arr.tolist(), [64, 65, 66, 67, 68, 69, 70])
        arr = spread._covering_sequences_at(576860)
        self.assertListEqual(arr.tolist(), [82403, 82404])


    def test_mean_of_covering_seq(self):
        chrv = Chromosome('VL')
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        spread_c0 = spread.mean_of_covering_seq()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


    def test_spread_c0_weighted(self):
        chrv = Chromosome('VL')
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        spread_c0 = spread.weighted_covering_seq()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


class TestChromosome(unittest.TestCase):
    def test_get_chr_prediction(self):
        chrv = Chromosome('IX')
        predict_df = chrv._get_chr_prediction()
        self.assertCountEqual(predict_df.columns, ['Sequence #', 'Sequence', 'C0'])


