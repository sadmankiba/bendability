from helsep import HelicalSeparationCounter

import pandas as pd
import numpy as np

import unittest

class TestHelicalSeparationCounter(unittest.TestCase):
    def test_count_helical_separation(self):
        seq = 'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC'

        # Explanation 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        # helical = 0 + 0 + 1 = 1
        # half-helical = 1 + 1 + 0 = 2
        # hs = 1 - 2 = -1
        self.assertEqual(HelicalSeparationCounter()._count_helical_separation(seq, ('TT', 'GC')), -1)
    

    def test_count_normalized_helical_separation(self):
        seq = 'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC'
        
        
        helc = HelicalSeparationCounter()
        # expected_dist_df =  pd.read_csv(self._expected_dist_file, sep='\t')
        expected_dist_df = helc.count_dist_random_seq()
        pair_expected_dist = expected_dist_df.loc[
            expected_dist_df['Pair'] == 'GC-TT'
        ].to_numpy().ravel()

        # Normalize dist
        helical = (np.array([1,0,1]) / pair_expected_dist[28:31]).max()
        half_helical = (np.array([1,1,1]) / pair_expected_dist[3:6]).max() \
            + (np.array([0, 1, 0]) / pair_expected_dist[13:16]).max()

        self.assertAlmostEqual(
            helc._count_normalized_helical_separation(seq, ('GC', 'TT')), 
            helical - half_helical
        )
        

    def test_find_helical_separation(self):

        df = pd.DataFrame({'Sequence': [
            'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACCGCCGGGCGCTCTC',
            'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC']})

        # Explanation 
        # First 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 34, 39, 41, 8, 4, 5, 24, 29, 31]
        # helical = 0 + 0 + 1 = 1
        # half-helical = 1 + 1 + 1 = 3
        # hs = 1 - 3 = -2
        # Second - same as last
        
        df_hel = HelicalSeparationCounter().find_helical_separation(df)
        self.assertGreater(len(df_hel.columns), len(df.columns))
        self.assertEqual(len(df_hel.columns.tolist()), 1 + 120 + 16)
        # Needs to be normalized
        # self.assertListEqual(df_hel['GC-TT'].tolist(), [-2, -1])


if __name__ == "__main__":
    unittest.main()