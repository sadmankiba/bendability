from data_organizer import ShapeOrganizerFactory, DataOrganizer, \
        BorutaFeatureSelector, ManualFeatureSelector

import numpy as np
import pandas as pd

import unittest

class TestDataOrganizer(unittest.TestCase):
    def setUp(self): 
        pass


    def test_one_hot_encode_shape(self):
        factory = ShapeOrganizerFactory('ohe', '')
        ohe_shape_encoder = factory.make_shape_organizer(None)
        enc_arr = ohe_shape_encoder._encode_shape(np.array([[3,7,2],[5,1,4]]), 3)
        print('enc_arr\n', enc_arr)
        self.assertTupleEqual(enc_arr.shape, (2,3,3))


    def test_classify(self):
        organizer = DataOrganizer(None, None, None, None, np.array([0.2, 0.6, 0.2]), None)
        df = pd.DataFrame({'C0': np.array([3,9,13,2,8,4,11])})
        cls = organizer._classify(df)
        self.assertListEqual(cls['C0'].tolist(), [1, 1, 2, 0, 1, 1, 2])


    def test_get_binary_classification(self):
        organizer = DataOrganizer(None, None, None, None, None, None) 
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df = organizer._get_binary_classification(df)
        self.assertListEqual(df['C0'].tolist(), [1, 0, 1])


    def test_get_balanced_classes(self):
        organizer = DataOrganizer(None, None, None, None, None, None) 
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df, _ = organizer._get_balanced_classes(df, df['C0'].to_numpy())
        self.assertCountEqual(df['C0'].tolist(), [0,0,0,1,1,1,2,2,2])
    

    def test_select_feat_boruta(self):
        X = np.array([  [0, 1, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 0, 1],
                        [1, 0, 1]])

        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        selector = BorutaFeatureSelector()
        selector.fit(X, y)
        self.assertListEqual(selector.support_.tolist(), [False, True, True])
        self.assertListEqual(selector.ranking_.tolist(), [2, 1, 1])

        X_sel = selector.transform(X)
        self.assertTupleEqual(X_sel.shape, (8, 2))
        

    def test_manual_feat_select(self):
        X = np.array([  [11, 3, 10],
                        [12, 4, 9],
                        [13, 5, 8],
                        [14, 8, 5],
                        [15, 9, 4],
                        [16, 10, 3],
                        [17, 7, 7],
                        [18, 6, 6]])

        y = np.array([0, 0, 0, 2, 2, 2, 1, 1])

        selector = ManualFeatureSelector()
        selector.fit(X, y)
        self.assertListEqual(selector.support_.tolist(), [False, True, True])
        self.assertListEqual(selector.ranking_.tolist(), [2, 1, 1])

        X_sel = selector.transform(X)
        self.assertTupleEqual(X_sel.shape, (8, 2))
        

if __name__ == '__main__':
    unittest.main()