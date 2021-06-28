from model import Model

import numpy as np

import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        pass


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
        
        model = Model(None, '', -1, -1, None)
        X_sel, sel_cols, col_rank = model._select_feat(X, y)
        self.assertTupleEqual(X_sel.shape, (8, 2))
        self.assertListEqual(sel_cols.tolist(), [False, True, True])
        self.assertListEqual(col_rank.tolist(), [2, 1, 1])

if __name__ == '__main__':
    unittest.main()