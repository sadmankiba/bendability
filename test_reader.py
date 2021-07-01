from reader import DNASequenceReader

import pandas as pd

import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        pass

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


if __name__ == '__main__':
    unittest.main()