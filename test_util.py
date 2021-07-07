from util import get_possible_seq, cut_sequence, HelicalSeparationCounter, \
        reverse_compliment_of, append_reverse_compliment

import pandas as pd
import numpy as np

import unittest


# https://stackoverflow.com/a/31832447/7283201

class TestUtil(unittest.TestCase):
    def setUp(self):
        pass


    def test_reverse_compliment_of(self):
        res = reverse_compliment_of('ATGCTAAC')
        # self.assertEqual(res, 'TACGATTG')
        self.assertEqual(res, 'GTTAGCAT')
    

    def test_append_reverse_compliment(self):
        df = pd.DataFrame({'Sequence': ['ATGCCGT', 'GCGATGC'], 'Col2': [5, 6]})
        rdf = append_reverse_compliment(df)

        self.assertGreater(len(rdf), len(df))
        self.assertCountEqual(rdf['Sequence'].tolist(), 
            ['ATGCCGT', 'GCGATGC', 'ACGGCAT', 'GCATCGC'])
        self.assertCountEqual(rdf['Col2'].tolist(), [5, 6, 5, 6])


    def test_get_possible_seq_two(self):
        possib_seq = get_possible_seq(size=2)
        expected = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', \
            'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']

        # Test two list have same content without regard to their order
        self.assertCountEqual(possib_seq, expected)
    

    def test_count_dist_random_seq(self):
        df = HelicalSeparationCounter().count_dist_random_seq()
        self.assertEqual(df.shape, (136, 49))


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
        

    def test_cut_sequence(self):
        df = pd.DataFrame({'Sequence': ['abcde', 'fghij']})
        cdf = cut_sequence(df, 2, 4)
        cdf_seq_list = cdf['Sequence'].tolist()
        expected = ['bcd', 'ghi']

        # Test two list have same content, also regarding their order
        self.assertListEqual(cdf_seq_list, expected)


if __name__ == "__main__":
    unittest.main()