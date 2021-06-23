from util import get_possible_seq, find_occurence, cut_sequence

import pandas as pd

import unittest


# https://stackoverflow.com/a/31832447/7283201

class TestListElements(unittest.TestCase):
    def setUp(self):
        self.expected = ['foo', 'bar', 'baz']
        self.result = ['baz', 'foo', 'bar']

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

    def test_cut_sequence(self):
        df = pd.DataFrame({'Sequence': ['abcde', 'fghij']})
        cdf = cut_sequence(df, 2, 4)
        print('cdf\n', cdf)
        cdf_seq_list = cdf['Sequence'].tolist()
        expected = ['bcd', 'ghi']

        # Test two list have same content, also regarding their order
        self.assertListEqual(cdf_seq_list, expected)

        
    

if __name__ == "__main__":
    unittest.main()