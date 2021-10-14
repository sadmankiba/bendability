import unittest
import time

import numpy as np

from chromosome.chromosome import Chromosome, ChrmCalc, Spread
from util.constants import CHRVL_LEN, CHRV_TOTAL_BP
from models.prediction import Prediction

class TestChrmCalc(unittest.TestCase):
    def test_moving_avg(self):
        arr = np.array([4, 6, 1, -9, 2, 7, 3])
        ma = ChrmCalc.moving_avg(arr, 4)
        assert ma.tolist() == [0.5, 0, 0.25, 0.75]

    def test_get_total_bp(self):
        assert ChrmCalc.total_bp(5) == 78


class TestSpread(unittest.TestCase):
    def test_covering_sequences_at(self):
        chrv = Chromosome('VL', None)
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        arr = spread._covering_sequences_at(30)
        self.assertListEqual(arr.tolist(), [1, 2, 3, 4, 5])
        arr = spread._covering_sequences_at(485)
        self.assertListEqual(arr.tolist(), [64, 65, 66, 67, 68, 69, 70])
        arr = spread._covering_sequences_at(576860)
        self.assertListEqual(arr.tolist(), [82403, 82404])

    def test_mean_of_7(self):
        chrv = Chromosome('VL', None)
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        spread_c0 = spread._mean_of_7()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP, ))

        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)

    def test_mean_of_covering_seq(self):
        chrv = Chromosome('VL', None)
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        spread_c0 = spread._mean_of_covering_seq()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP, ))

        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)

    def test_spread_c0_weighted(self):
        chrv = Chromosome('VL', None)
        spread = Spread(chrv._df['C0'].to_numpy(), chrv._chr_id)
        spread_c0 = spread._weighted_covering_seq()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP, ))

        samples = spread_c0[np.random.randint(0, CHRVL_LEN - 1, 100)]
        self.assertTrue(np.all(samples < 2.5))
        self.assertTrue(np.all(samples > -2.5))


class TestChromosome:
    def test_get_chr_prediction(self):
        chrm = Chromosome('IX', Prediction(model=6))
        predict_df = chrm._get_chrm_df()
        assert predict_df.columns.tolist() == ['Sequence #', 'Sequence', 'C0']
    
    def test_without_prediction_initialization(self):
        chrm = Chromosome('IX')
        assert chrm._df.columns.tolist() == ['Sequence #', 'Sequence']
    
    def test_mean_c0_around_bps(self):
        chrm = Chromosome('XI', Prediction())
        mean = chrm.mean_c0_around_bps([5000, 10000, 12000], 60, 40)
        assert mean.shape == (60 + 40 + 1, )
    
    def test_mean_c0_of_segments(self):
        mn = Chromosome('VL').mean_c0_of_segments([5000, 8000], 100, 50)
        assert -2 < mn < 2
    
    def test_mean_c0_at_bps(self):
        mn = Chromosome('IX', Prediction(30)).mean_c0_at_bps([12000, 20000], 200, 200)
        assert mn[0] > -1
        assert mn[0] < 1
        assert mn[1] > -1
        assert mn[1] < 1
    
    def test_get_spread_saving(self):
        chrm = Chromosome('VIII', Prediction(6))
        t = time.time()
        sp_one = chrm.get_spread()
        dr_one = time.time() - t
        assert hasattr(chrm, '_c0_spread')
        
        t = time.time()
        sp_two = chrm.get_spread()
        dr_two = time.time() - t
        assert dr_two < dr_one * 0.01