from feature_model.data_organizer import (
    DataOrganizeOptions,
    ShapeOrganizerFactory,
    DataOrganizer,
    ClassificationMaker,
    TrainTestSequenceLibraries,
    SequenceLibrary,
)
from feature_model.feat_selector import FeatureSelectorFactory
from util.reader import DNASequenceReader
from util.constants import CNL, CNL_LEN, RL, TL, TL_LEN, RL_LEN
from util.util import PathUtil

import numpy as np
import pandas as pd
import pytest

import unittest
from pathlib import Path
import itertools as it


class TestDataOrganizer(unittest.TestCase):
    def setUp(self):
        pass

    @pytest.mark.skip(reason="Random selection code in method raises error")
    def test_get_seq_train_test(self):
        libraries: TrainTestSequenceLibraries = {
            "train": [
                SequenceLibrary(name=TL, quantity=50000),
                SequenceLibrary(name=CNL, quantity=15000),
            ],
            "test": [SequenceLibrary(name=RL, quantity=10000)],
            "train_test": [],
            "seq_start_pos": 1,
            "seq_end_pos": 50,
        }
        options = DataOrganizeOptions(k_list=[2, 3])
        feature_factory = FeatureSelectorFactory("all")
        selector = feature_factory.make_feature_selector()
        data_organizer = DataOrganizer(libraries, None, selector, options)
        X_train, X_test, y_train, y_test = data_organizer.get_seq_train_test(
            classify=False
        )

        reader = DNASequenceReader()
        all_df = reader.get_processed_data()

        assert X_train.shape[0] == 65000
        assert y_train.shape[0] == 65000
        assert X_test.shape[0] == 10000
        assert y_test.shape[0] == 10000

    def test_get_helical_sep(self):
        libraries: TrainTestSequenceLibraries = {
            "train": [SequenceLibrary(name=CNL, quantity=CNL_LEN)],
            "test": [SequenceLibrary(name=RL, quantity=RL_LEN)],
            "train_test": [],
            "seq_start_pos": 1,
            "seq_end_pos": 50,
        }

        organizer = DataOrganizer(libraries, None, None)
        hel_dfs = organizer._get_helical_sep()
        self.assertEqual(len(hel_dfs["train"][0].columns), 3 + 120 + 16)
        self.assertEqual(len(hel_dfs["test"][0].columns), 3 + 120 + 16)

        saved_train_file = Path(
            f"{PathUtil.get_data_dir()}/generated_data/helical_separation"
            f"/{libraries['train'][0]['name']}_{libraries['seq_start_pos']}_{libraries['seq_end_pos']}_hs.tsv"
        )

        saved_test_file = Path(
            f"{PathUtil.get_data_dir()}/generated_data/helical_separation"
            f"/{libraries['test'][0]['name']}_{libraries['seq_start_pos']}_{libraries['seq_end_pos']}_hs.tsv"
        )

        self.assertEqual(saved_train_file.is_file(), True)
        self.assertEqual(saved_test_file.is_file(), True)

    def test_get_kmer_count(self):
        libraries: TrainTestSequenceLibraries = {
            "train": [SequenceLibrary(name=TL, quantity=TL_LEN)],
            "test": [SequenceLibrary(name=RL, quantity=RL_LEN)],
            "train_test": [],
            "seq_start_pos": 1,
            "seq_end_pos": 50,
        }

        k_list = [2, 3]
        options = DataOrganizeOptions(k_list=k_list)

        organizer = DataOrganizer(libraries, None, None, options)
        kmer_dfs = organizer._get_kmer_count()
        self.assertEqual(len(kmer_dfs["train"][0].columns), 3 + 4 ** 2 + 4 ** 3)
        self.assertEqual(len(kmer_dfs["test"][0].columns), 3 + 4 ** 2 + 4 ** 3)

        for lib_type, k in it.product(["train", "test"], k_list):
            saved_file = Path(
                f"../data/generated_data/kmer_count"
                f"/{libraries[lib_type][0]['name']}_{libraries['seq_start_pos']}"
                f"_{libraries['seq_end_pos']}_kmercount_{k}.tsv"
            )

            self.assertEqual(saved_file.is_file(), True)

    def test_one_hot_encode_shape(self):
        factory = ShapeOrganizerFactory("ohe", "")
        ohe_shape_encoder = factory.make_shape_organizer(None)
        enc_arr = ohe_shape_encoder._encode_shape(np.array([[3, 7, 2], [5, 1, 4]]), 3)

        self.assertTupleEqual(enc_arr.shape, (2, 3, 3))

        expected = [
            [[1, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 0, 0]],
        ]
        self.assertListEqual(enc_arr.tolist(), expected)

    def test_classify(self):
        class_maker = ClassificationMaker(np.array([0.2, 0.6, 0.2]), False)
        df = pd.DataFrame({"C0": np.array([3, 9, 13, 2, 8, 4, 11])})
        cls = class_maker.classify(df)
        self.assertListEqual(cls["C0"].tolist(), [1, 1, 2, 0, 1, 1, 2])

    def test_get_binary_classification(self):
        class_maker = ClassificationMaker(None, None)
        df = pd.DataFrame({"C0": [1, 2, 0, 1, 1, 2]})
        df = class_maker._get_binary_classification(df)
        self.assertListEqual(df["C0"].tolist(), [1, 0, 1])

    def test_get_balanced_classes(self):
        class_maker = ClassificationMaker(None, None)
        df = pd.DataFrame({"C0": [1, 2, 0, 1, 1, 2]})
        df, _ = class_maker.get_balanced_classes(df, df["C0"].to_numpy())
        self.assertCountEqual(df["C0"].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
