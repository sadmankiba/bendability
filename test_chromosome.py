from chromosome import Chromosome, ChromosomeUtil

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


class TestChromosome(unittest.TestCase):
    def test_covering_sequences_at(self):
        chrv = Chromosome('VL')
        arr = chrv._covering_sequences_at(30)
        self.assertListEqual(arr.tolist(), [1, 2, 3, 4, 5])
        arr = chrv._covering_sequences_at(485)
        self.assertListEqual(arr.tolist(), [64, 65, 66, 67, 68, 69, 70])
        arr = chrv._covering_sequences_at(576860)
        self.assertListEqual(arr.tolist(), [82403, 82404])
    
    
    def test_plot_c0_vs_dist_from_dyad_spread(self):
        chrv = Chromosome('VL')
        chrv.plot_c0_vs_dist_from_dyad_spread(150)
        path = Path('figures/chrv/c0_dyad_dist_150_balanced.png')
        self.assertTrue(path.is_file())
    

    def test_spread_c0_balanced(self):
        chrv = Chromosome('VL')
        spread_c0 = chrv.spread_c0_balanced()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


    def test_spread_c0_weighted(self):
        chrv = Chromosome('VL')
        spread_c0 = chrv.spread_c0_weighted()
        self.assertTupleEqual(spread_c0.shape, (CHRV_TOTAL_BP,))
        
        samples = spread_c0[np.random.randint(0,CHRVL_LEN - 1,100)]
        self.assertTrue(np.all(samples) < 2.5)
        self.assertTrue(np.all(samples) > -2.5)


    def test_read_chr_prediction(self):
        chrv = Chromosome('VL')
        predict_df = chrv._read_chr_prediction('V')
        self.assertCountEqual(predict_df.columns, ['Sequence #', 'Sequence', 'C0'])


    def test_get_nuc_occupancy(self):
        chr = Chromosome('VII')
        nuc_occ = chr.get_nucleosome_occupancy()
        assert nuc_occ.shape == (chr._total_bp, )
        assert any(nuc_occ)

