from feat_selector import BorutaFeatureSelector, ManualFeatureSelector, FeatureSelectorFactory

import unittest
import numpy as np

class TestFeatSelector(unittest.TestCase):
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