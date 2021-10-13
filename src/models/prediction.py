from __future__ import annotations
import inspect
from typing import Union

import keras
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from .cnnmodel import CNNModel6, CNNModel30
from .data_preprocess import Preprocess
from .parameters import get_parameters
from util.custom_types import LIBRARY_NAMES
from util.reader import DNASequenceReader
from util.util import IOUtil, PathUtil


class Prediction:
    def __init__(self, model: Union[int, keras.Model] = 6):
        if isinstance(model, int): 
            self._model_no = model
            self._model = self._load_model()
        elif isinstance(model, keras.Model):
            self._model = model
    
    def __str__(self):
        return str(self._model_no)

    def _select_model(self) -> tuple[CNNModel6, str, str]:
        parent_dir = PathUtil.get_parent_dir(inspect.currentframe())

        if self._model_no == 6:
            return (CNNModel6, f'{parent_dir}/parameter_model6.txt',
                    f'{parent_dir}/model_weights/w6.h5_archived')
        elif self._model_no == 30:
            return (
                CNNModel30,
                f'{parent_dir}/parameter_model30.txt',
                f'{parent_dir}/model_weights/w30.h5'
            )

    def _load_model(self) -> keras.Model:
        nn_model, parameter_file, weight_file = self._select_model()
        params = get_parameters(parameter_file)
        dim_num = (-1, 50, 4)
        
        nn: CNNModel6 = nn_model(dim_num=dim_num, **params)
        model = nn.create_model()
        model.load_weights(weight_file)
        return model

    # TODO: Use CedricFR/dataenforce for DF type hints
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        y_pred = self._model.predict({
            'forward': data["forward"], 
            'reverse': data["reverse"]
        }).flatten()
        return df.assign(c0_predict=y_pred)

    def predict_lib(self, lib: LIBRARY_NAMES) -> pd.DataFrame:
        predict_df = self.predict(
            DNASequenceReader().get_processed_data()[lib])
        IOUtil().save_tsv(
            predict_df,
            f'{PathUtil.get_data_dir()}/generated_data/predictions/{lib}_pred_m_{self._model_no}.tsv'
        )
        return predict_df

    def predict_metrics_lib(self, lib: LIBRARY_NAMES) -> None:
        pred_df = self.predict_lib(lib)
        y = pred_df['C0'].to_numpy()
        y_pred = self.predict(pred_df)['c0_predict'].to_numpy()
        assert y_pred.shape == y.shape

        metrics_df = pd.DataFrame({
            'library': [lib],
            'r2_score': [r2_score(y, y_pred)],
            'pearsons_corr': [pearsonr(y, y_pred)[0]],
            'spearmans_corr': [spearmanr(y, y_pred)[0]]
        })
        IOUtil().append_tsv(
            metrics_df,
            f'{PathUtil.get_data_dir()}/generated_data/prediction_metrics/pred_m_{self._model_no}.tsv'
        )
