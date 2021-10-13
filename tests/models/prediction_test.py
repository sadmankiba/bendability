from numpy.testing._private.utils import assert_almost_equal
from models.prediction import Prediction
from util.constants import RL, CNL
from util.reader import DNASequenceReader
from util.util import PathUtil

import keras
import numpy as np

from pathlib import Path

# TODO : Unit tests should finish fast

class TestPrediction:
    def test_load_model(self):
        pred = Prediction()
        assert isinstance(pred._model, keras.Model)
    
    def test_predict(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        result_df = Prediction().predict(df)
        assert 'c0_predict' in set(result_df.columns)
    
    def test_predict_model6(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:10]
        result_df = Prediction(6).predict(df)
        
        assert_almost_equal(np.round(result_df['c0_predict'], 3).tolist(), 
            [ 0.122, -0.274,  0.606,  0.355,  0.106, 
             -0.411, -0.993, -0.728, -0.461,  0.295], decimal=3)

    def test_predict_model30(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:10]
        result_df = Prediction(30).predict(df)
        
        assert_almost_equal(np.round(result_df['c0_predict'], 3).tolist(), 
            [-0.013, -0.465, 0.531, 0.204, 0.241, 
            -0.465, -1.284, -0.819, -0.314, 0.283], decimal=3)
            

    def test_predict_lib(self):
        Prediction().predict_lib(RL)
        assert Path(
            f'{PathUtil.get_data_dir()}/generated_data/predictions/{RL}_pred_m_6.tsv').is_file()

    def test_predict_metrics_lib(self):
        Prediction().predict_metrics_lib(RL)
        assert Path(
            f'{PathUtil.get_data_dir()}/generated_data/prediction_metrics/pred_m_6.tsv').is_file()

    
    def test_predict_metrics_lib_m30(self):
        Prediction(model_no=30).predict_metrics_lib(RL)
        assert Path(
            f'{PathUtil.get_data_dir()}/generated_data/prediction_metrics/pred_m_30.tsv').is_file()
