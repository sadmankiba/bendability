from helsep import HelicalSeparationCounter

import pandas as pd
import numpy as np

import unittest

class TestHelicalSeparationCounter(unittest.TestCase):
    def test_get_all_dist(self):
        seq = 'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC'
        # Explanation 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        helsep = HelicalSeparationCounter()
        all_dist = helsep._get_all_dist(seq)
        pair_idx = helsep._dinc_pairs.index(('GC', 'TT'))
        p_expected = np.bincount([2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31], minlength=49)[1:]
        
        self.assertListEqual(all_dist[pair_idx].tolist(), p_expected.tolist())
    

    def test_normalized_helical_sep_of(self):
        seq = 'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC'
        # Explanation 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        # helical = max((0/n,0/n,0/n)) + max((0/n,0/n,0/n)) + max(1/n,0/n,1/n)
        # half-helical = max((1/n,1/n,1/n)) + max((0/n,1/n,0/n)) + max(0/n,0/n,0/n) 
        # hs = h -hh
        
        helsep = HelicalSeparationCounter()
        expected_dist = helsep.calculate_expected_p().values
        pair_idx = helsep._dinc_pairs.index(('GC', 'TT'))

        # Normalize dist
        helical = (np.array([1,0,1]) / expected_dist[pair_idx, 28:31]).max()
        half_helical = (np.array([1,1,1]) / expected_dist[pair_idx, 3:6]).max() \
            + (np.array([0, 1, 0]) / expected_dist[pair_idx, 13:16]).max()

        hs = helsep._normalized_helical_sep_of([seq])
        self.assertTupleEqual(hs.shape, (1,136))
        self.assertAlmostEqual(hs[0, pair_idx], helical - half_helical)
        

    def test_find_helical_separation(self):

        df = pd.DataFrame({'Sequence': [
            'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACCGCCGGGCGCTCTC',
            'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC']})

        # Explanation 
        # First 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 34, 39, 41, 8, 4, 5, 24, 29, 31]
        # Second - same as last
        
        df_hel = HelicalSeparationCounter().find_helical_separation(df)
        self.assertGreater(len(df_hel.columns), len(df.columns))
        self.assertEqual(len(df_hel.columns.tolist()), 1 + 120 + 16)
        # Needs to be normalized
        # self.assertListEqual(df_hel['GC-TT'].tolist(), [-2, -1])


if __name__ == "__main__":
    unittest.main()