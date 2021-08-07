from evaluation import Evaluation

# Import from parent directory
import sys
import os

from reader import DNASequenceReader

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from constants import CNL

from pathlib import Path


class TestEvaluation:
    def test_check_performance(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        Evaluation().check_performance(df)
        assert True

    def test_predict(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        result_df = Evaluation().predict(df)
        assert 'c0_predict' in set(result_df.columns)
