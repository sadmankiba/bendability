from __future__ import annotations

from meuseum_mod.model6 import nn_model as nn_model6
from DNABendabilityModels.meuseum_modifications.models.model30 import nn_model as nn_model30
from meuseum_mod.data_preprocess import Preprocess
from custom_types import LIBRARY_NAMES
from reader import DNASequenceReader
from util import IOUtil

import keras
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from pathlib import Path
import inspect


def get_parameters(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split()
            dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])

    return dict


class Prediction:
    def __init__(self, model_no: int = 6):
        self._model_no = model_no
        self._model = self._load_model()
    
    def __str__(self):
        return str(self._model_no)

    def _select_model(self) -> tuple[nn_model6, str, str]:
        if self._model_no == 6:
            return (nn_model6, 'meuseum_mod/parameter1.txt',
                    'meuseum_mod/model_weights/w6.h5_archived')
        elif self._model_no == 30:
            return (
                nn_model30,
                'DNABendabilityModels/meuseum_modifications/parameter1.txt',
                'DNABendabilityModels/meuseum_modifications/model_weights/w30.h5'
            )

    def _load_model(self) -> keras.Model:
        # Find parent directory path dynamically
        parent_dir = Path(inspect.getabsfile(inspect.currentframe())).parent

        nn_model, parameter_file, weight_file = self._select_model()

        params = get_parameters(parameter_file)

        dim_num = (-1, 50, 4)

        print('Initializing nn_model object...')
        nn = nn_model(dim_num=dim_num,
                      filters=params["filters"],
                      kernel_size=params["kernel_size"],
                      pool_type=params["pool_type"],
                      regularizer=params["regularizer"],
                      activation_type=params["activation_type"],
                      epochs=params["epochs"],
                      batch_size=params["batch_size"],
                      loss_func=params["loss_func"],
                      optimizer=params["optimizer"])

        print('Creating Keras model...')
        model = nn.create_model()

        print('Loading weights in model...')
        model.load_weights(weight_file)
        return model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        x1 = data["forward"]
        x2 = data["reverse"]

        y_pred = self._model.predict({'forward': x1, 'reverse': x2}).flatten()
        return df.assign(c0_predict=y_pred)

    def predict_lib(self, lib: LIBRARY_NAMES) -> pd.DataFrame:
        predict_df = self.predict(
            DNASequenceReader().get_processed_data()[lib])
        IOUtil().save_tsv(
            predict_df,
            f'data/generated_data/predictions/{lib}_pred_m_{self._model_no}.tsv'
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
            f'data/generated_data/prediction_metrics/pred_m_{self._model_no}.tsv'
        )
