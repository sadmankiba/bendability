from prediction import Prediction
from constants import RL, CNL
from reader import DNASequenceReader

from pathlib import Path


class TestPrediction:
    def test_predict(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        result_df = Prediction().predict(df)
        assert 'c0_predict' in set(result_df.columns)

    def test_predict_lib(self):
        Prediction().predict_lib(RL)
        assert Path(f'data/generated_data/predictions/{RL}_pred_m_6.tsv').is_file()
    
    def test_predict_metrics_lib(self):
        Prediction().predict_metrics_lib(RL)
        assert Path(f'data/generated_data/prediction_metrics/pred_m_6.tsv').is_file()
    
    def test_predict_metrics_lib_m30(self):
        Prediction(model_no=30).predict_metrics_lib(RL)
        assert Path(f'data/generated_data/prediction_metrics/pred_m_30.tsv').is_file()
    
    
