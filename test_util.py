from util import get_possible_seq, cut_sequence, \
        reverse_compliment_of, append_reverse_compliment, IOUtil

import pandas as pd
import numpy as np

import unittest
from pathlib import Path
import os

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

    def test_cut_sequence(self):
        df = pd.DataFrame({'Sequence': ['abcde', 'fghij']})
        cdf = cut_sequence(df, 2, 4)
        cdf_seq_list = cdf['Sequence'].tolist()
        expected = ['bcd', 'ghi']

        # Test two list have same content, also regarding their order
        self.assertListEqual(cdf_seq_list, expected)


class TestIOUtil:
    def test_append_tsv(self):
        sample_df = pd.DataFrame({'a': [1, 2], 'b': [11, 12], 'c': [21,25]})
        path = Path('data/generated_data/test_append.tsv')

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)

        sample_df.to_csv(path, sep='\t', index=False)
        other_df = pd.DataFrame({'a': [3], 'b': [13], 'd': [31]})

        IOUtil().append_tsv(other_df, path)
        append_df = pd.read_csv(path, sep='\t')
        assert len(append_df) == 3
        assert len(append_df.columns) == 4
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
