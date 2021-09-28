from models.evaluation import Evaluation
from util.reader import DNASequenceReader
from util.constants import CNL

class TestEvaluation:
    def test_check_performance(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        Evaluation().check_performance(df)
        assert True

    def test_predict(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        result_df = Evaluation().predict(df)
        assert 'c0_predict' in set(result_df.columns)
