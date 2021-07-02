from util import get_possible_seq, find_occurence, cut_sequence, \
    find_occurence_individual, count_helical_separation, find_helical_separation, \
        reverse_compliment_of, append_reverse_compliment

import pandas as pd

import unittest


# https://stackoverflow.com/a/31832447/7283201

class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_reverse_compliment_of(self):
        res = reverse_compliment_of('ATGCTAAC')
        self.assertEqual(res, 'TACGATTG')
    
    def test_append_reverse_compliment(self):
        df = pd.DataFrame({'Sequence': ['ATGCCGT', 'GCGATGC'], 'K': [5, 6]})
        rdf = append_reverse_compliment(df)

        self.assertGreater(len(rdf), len(df))
        self.assertCountEqual(rdf['Sequence'].tolist(), ['ATGCCGT', 'GCGATGC', 'TACGGCA', 'CGCTACG'])
        self.assertCountEqual(rdf['K'].tolist(), [5, 6, 5, 6])

    def test_get_possible_seq_two(self):
        possib_seq = get_possible_seq(size=2)
        expected = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', \
            'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']

        # Test two list have same content without regard to their order
        self.assertCountEqual(possib_seq, expected)
    

    def test_find_occurence(self):
        seq_list = ['AGTTC', 'GATCC']
        occur_dict = find_occurence(seq_list, unit_size=2)
        expected = {'AA': 0, 'AT': 1, 'AG': 1, 'AC': 0, 'TA': 0, 'TT': 1, 'TG': 0, 'TC': 2, \
            'GA': 1, 'GT': 1, 'GG': 0, 'GC': 0, 'CA': 0, 'CT': 0, 'CG': 0, 'CC': 1}
        
        self.assertDictEqual(occur_dict, expected)

    
    def test_find_occurence_individual(self):
        df = pd.DataFrame({'Sequence': ['ACGT', 'AAGT', 'CTAG']})
        df_occur = find_occurence_individual(df, [2])

        self.assertGreater(len(df_occur.columns), len(df.columns))
        self.assertListEqual(df_occur['AA'].tolist(), [0, 1, 0])
        self.assertListEqual(df_occur['AG'].tolist(), [0, 1, 1])


    def test_count_helical_separation(self):
        seq = 'AAATTGCCTGCTCTTCCTGCGACCAGTCCTCTCGACGCCCGGGCGCTCTC'

        # Explanation 
        # TT -> [3, 13]
        # GC -> [5, 9, 18, 36, 42, 44]
        # Absolute diff -> [2, 6, 15, 33, 39, 41, 8, 4, 5, 23, 29, 31]
        # helical = 0 + 0 + 1 = 1
        # half-helical = 1 + 1 + 0 = 2
        # hs = 1 - 2 = -1
        self.assertEqual(count_helical_separation(seq, ('TT', 'GC')), -1)
        

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
        df_hel = find_helical_separation(df)
        self.assertGreater(len(df_hel.columns), len(df.columns))
        self.assertEqual(len(df_hel.columns.tolist()), 1 + 120 + 16)
        self.assertListEqual(df_hel['GC-TT'].tolist(), [-2, -1])
        

    def test_cut_sequence(self):
        df = pd.DataFrame({'Sequence': ['abcde', 'fghij']})
        cdf = cut_sequence(df, 2, 4)
        cdf_seq_list = cdf['Sequence'].tolist()
        expected = ['bcd', 'ghi']

        # Test two list have same content, also regarding their order
        self.assertListEqual(cdf_seq_list, expected)


if __name__ == "__main__":
    unittest.main()