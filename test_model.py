from model import one_hot_encode_shape, classify_arr, get_binary_classification, \
    get_balanced_classes, Model

import numpy as np
import pandas as pd

import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_one_hot_encode_shape(self):
        arr = np.array([[3,7,2],[5,1,4]])
        enc_arr = one_hot_encode_shape(arr, '', 3)
        print('enc_arr\n', enc_arr)
        self.assertTupleEqual(enc_arr.shape, (2,3,3))


    def test_classify_arr(self):
        arr = np.array([3,9,13,2,8,4,11])
        cls = classify_arr(arr, np.array([0.2, 0.6, 0.2]))
        self.assertListEqual(cls.tolist(), [1, 1, 2, 0, 1, 1, 2])

    def test_get_binary_classification(self):
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df = get_binary_classification(df)
        self.assertListEqual(df['C0'].tolist(), [1, 0, 1])

    def test_get_balanced_classes(self):
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df, _ = get_balanced_classes(df, df['C0'].to_numpy())
        self.assertCountEqual(df['C0'].tolist(), [0,0,0,1,1,1,2,2,2])

    def test_select_feat(self):
        X = np.array([  [0, 1, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 0, 1],
                        [1, 0, 1]])

        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        model = Model('', -1, -1)
        X_sel, sel_cols, col_rank = model._select_feat(X, y)
        self.assertTupleEqual(X_sel.shape, (8, 2))
        self.assertListEqual(sel_cols.tolist(), [False, True, True])
        self.assertListEqual(col_rank.tolist(), [2, 1, 1])

if __name__ == '__main__':
    unittest.main()