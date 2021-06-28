from data_organizer import ShapeEncoderFactory, DataOrganizer

import numpy as np
import pandas as pd

import unittest

class TestDataOrganizer(unittest.TestCase):
    def setUp(self): 
        pass


    def test_one_hot_encode_shape(self):
        factory = ShapeEncoderFactory()
        ohe_shape_encoder = factory.make_shape_encoder('ohe', '')
        enc_arr = ohe_shape_encoder.encode_shape(np.array([[3,7,2],[5,1,4]]), 3)
        print('enc_arr\n', enc_arr)
        self.assertTupleEqual(enc_arr.shape, (2,3,3))


    def test_classify(self):
        organizer = DataOrganizer(np.array([0.2, 0.6, 0.2]), None, None)
        df = pd.DataFrame({'C0': np.array([3,9,13,2,8,4,11])})
        cls = organizer.classify(df) 
        self.assertListEqual(cls['C0'].tolist(), [1, 1, 2, 0, 1, 1, 2])


    def test_get_binary_classification(self):
        organizer = DataOrganizer(np.empty(0), None, None) 
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df = organizer._get_binary_classification(df)
        self.assertListEqual(df['C0'].tolist(), [1, 0, 1])


    def test_get_balanced_classes(self):
        organizer = DataOrganizer(np.empty(0), None, None) 
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df, _ = organizer.get_balanced_classes(df, df['C0'].to_numpy())
        self.assertCountEqual(df['C0'].tolist(), [0,0,0,1,1,1,2,2,2])

if __name__ == '__main__':
    unittest.main()