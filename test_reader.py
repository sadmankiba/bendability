from constants import CHRVL
from reader import DNASequenceReader

import pandas as pd

import unittest

# TODO: Convert unittest to pytest

class TestModel(unittest.TestCase):
    def setUp(self):
        pass
    
    
    def test_get_raw_data_cnl(self):
        reader = DNASequenceReader()
        cnl_df_raw = reader._get_raw_data()[0]
        required_cols = ["Sequence #", "Sequence", " C0"]
        self.assertTrue(set(cnl_df_raw.columns).issuperset(required_cols))


    def test_get_raw_data_rl(self):
        reader = DNASequenceReader()
        rl_df_raw = reader._get_raw_data()[1]
        required_cols = ["Sequence #", "Sequence", " C0"]
        self.assertTrue(set(rl_df_raw.columns).issuperset(required_cols))
    

    def test_get_raw_data_tl(self):
        reader = DNASequenceReader()
        tl_df_raw = reader._get_raw_data()[2]
        required_cols = ["Sequence #", "Sequence", " C0"]
        self.assertTrue(set(tl_df_raw.columns).issuperset(required_cols))


    def test_get_raw_data_chrvl(self):
        reader = DNASequenceReader()
        chrvl_df_raw = reader._get_raw_data()[3]
        required_cols = ["Sequence #", "Sequence", " C0"]
        self.assertTrue(set(chrvl_df_raw.columns).issuperset(required_cols))


    def test_get_raw_data_libl(self):
        reader = DNASequenceReader()
        libl_df_raw = reader._get_raw_data()[4]
        required_cols = ["Sequence #", "Sequence", " C0"]
        self.assertTrue(set(libl_df_raw.columns).issuperset(required_cols))


    def test_preprocess(self):
        df = pd.DataFrame({'Sequence #': [1,2],
                        'Sequence': ['TTTCTTCACTTATCTCCCACCGTCCTCCGCACTTATGTACTGTGCTGAGATATAGTAGATTCTGCGTGTGATCGAGGCAGAAGACAAGGGAACGAAATAG', 
                                     'TTTCTTCACTTATCTCCCACCGTCCGTCTCGATCCACCGCTAGTAGTAAGACAACAGGGCTGCCTGGCTTCAACTGGCAGAAGACAAGGGAACGAAATAG'],
                        ' C0': [-1, 1]
                        })
        reader = DNASequenceReader()
        df = reader._preprocess(df)
        
        sequences = df['Sequence'].tolist()
        self.assertEqual(len(sequences[0]), 50)
        self.assertEqual(len(sequences[1]), 50)


    def test_read_nuc_center(self):
        nuc_df = DNASequenceReader().read_nuc_center()
        self.assertEqual(len(nuc_df), 67548) 


    def test_read_library_prediction(self):
        reader = DNASequenceReader()
        predict_df = reader.read_library_prediction(CHRVL)
        self.assertCountEqual(predict_df.columns, ['Sequence #', 'Sequence', 'C0'])


if __name__ == '__main__':
    unittest.main()