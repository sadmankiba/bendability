from data_organizer import DataOrganizeOptions, ShapeOrganizerFactory, DataOrganizer, \
        BorutaFeatureSelector, ManualFeatureSelector, SequenceLibrary, ClassificationMaker, \
            TrainTestSequenceLibraries
from reader import DNASequenceReader
from constants import CNL, RL, TL

import numpy as np
import pandas as pd

import unittest
from pathlib import Path
import itertools as it

class TestDataOrganizer(unittest.TestCase):
    def setUp(self): 
        pass

    def test_get_seq_train_test(self):
        libraries: TrainTestSequenceLibraries = {
            'train': [TL, CNL],
            'test': [RL], 
            'train_test': [], 
            'seq_start_pos': 1,
            'seq_end_pos': 50
        }
        options: DataOrganizeOptions = {
            'k_list': [2,3],
            'range_split': None,
            'binary_class': None, 
            'balance': None
        }
        data_organizer = DataOrganizer(libraries, None, None, options)
        X_train, X_test, y_train, y_test = data_organizer.get_seq_train_test(classify=False)

        reader = DNASequenceReader()
        all_df = reader.get_processed_data()
        
        train_total_len = sum(map(len, map(lambda name: all_df[name], libraries['train'])))
        test_total_len = sum(map(len, map(lambda name: all_df[name], libraries['test'])))

        assert X_train.shape[0] == train_total_len
        assert y_train.shape[0] == train_total_len
        assert X_test.shape[0] == test_total_len
        assert y_test.shape[0] == test_total_len

    def test_get_helical_sep(self):
        libraries: TrainTestSequenceLibraries = {
            'train': [CNL],
            'test': [TL], 
            'seq_start_pos': 1,
            'seq_end_pos': 50
        }

        organizer = DataOrganizer(libraries, None, None, None)
        hel_dfs = organizer._get_helical_sep()
        self.assertEqual(len(hel_dfs['train'][0].columns), 3 + 120 + 16)
        self.assertEqual(len(hel_dfs['test'][0].columns), 3 + 120 + 16)

        saved_train_file = Path(f"data/generated_data/helical_separation"
            f"/{libraries['train'][0]}_{libraries['seq_start_pos']}_{libraries['seq_end_pos']}_hs.tsv")
        
        saved_test_file = Path(f"data/generated_data/helical_separation"
            f"/{libraries['test'][0]}_{libraries['seq_start_pos']}_{libraries['seq_end_pos']}_hs.tsv")
        
        self.assertEqual(saved_train_file.is_file(), True)
        self.assertEqual(saved_test_file.is_file(), True)


    def test_get_kmer_count(self):
        libraries: TrainTestSequenceLibraries = {
            'train': [TL],
            'test': [RL], 
            'seq_start_pos': 1,
            'seq_end_pos': 50
        }

        k_list = [2, 3] 
        options: DataOrganizeOptions = {
            'k_list': k_list,
            'range_split': None,
            'binary_class': None, 
            'balance': None
        }

        organizer = DataOrganizer(libraries, None, None, options)
        kmer_dfs = organizer._get_kmer_count()
        self.assertEqual(len(kmer_dfs['train'][0].columns), 3 + 4**2 + 4**3)
        self.assertEqual(len(kmer_dfs['test'][0].columns), 3 + 4**2 + 4**3)

        for lib_type, k in it.product(['train', 'test'], k_list):
            saved_file = Path(f"data/generated_data/kmer_count"
                f"/{libraries[lib_type][0]}_{libraries['seq_start_pos']}"
                f"_{libraries['seq_end_pos']}_kmercount_{k}.tsv")
            
            self.assertEqual(saved_file.is_file(), True)
        

    def test_one_hot_encode_shape(self):
        factory = ShapeOrganizerFactory('ohe', '')
        ohe_shape_encoder = factory.make_shape_organizer(None)
        enc_arr = ohe_shape_encoder._encode_shape(np.array([[3,7,2],[5,1,4]]), 3)

        self.assertTupleEqual(enc_arr.shape, (2,3,3))
        
        expected = [[[1, 0, 1],
                     [0, 0, 0],
                     [0, 1, 0]], 
                     
                     [[0, 1, 0],
                     [1, 0, 1],
                     [0, 0, 0]]]
        self.assertListEqual(enc_arr.tolist(), expected)


    def test_classify(self):
        class_maker = ClassificationMaker(np.array([0.2, 0.6, 0.2]), False)
        df = pd.DataFrame({'C0': np.array([3,9,13,2,8,4,11])})
        cls = class_maker.classify(df)
        self.assertListEqual(cls['C0'].tolist(), [1, 1, 2, 0, 1, 1, 2])


    def test_get_binary_classification(self):
        class_maker = ClassificationMaker(None, None) 
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df = class_maker._get_binary_classification(df)
        self.assertListEqual(df['C0'].tolist(), [1, 0, 1])


    def test_get_balanced_classes(self):
        class_maker = ClassificationMaker(None, None)
        df = pd.DataFrame({'C0': [1, 2, 0, 1, 1, 2]})
        df, _ = class_maker.get_balanced_classes(df, df['C0'].to_numpy())
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