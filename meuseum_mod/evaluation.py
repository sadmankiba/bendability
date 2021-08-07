# Import from parent directory
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from constants import CNL
from reader import DNASequenceReader
from meuseum_mod.model6 import nn_model
from meuseum_mod.data_preprocess import Preprocess

import keras
from contextlib import suppress
from os import stat
import sys
import time
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logomaker as lm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from plotnine import ggplot, aes, xlim, ylim, stat_bin_2d

import shutil
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


def save_kernel_weights_logos(model):
    with open('kernel_weights/6', 'w') as f:
        for layer_num in range(2, 3):
            layer = model.layers[layer_num]
            config = layer.get_config()
            print(config, file=f)
            weights = layer.get_weights()
            w = tf.transpose(weights[0], [2, 0, 1])
            alpha = 20
            beta = 1 / alpha
            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(
                lambda x: tf.math.exp(
                    tf.subtract(
                        tf.subtract(
                            tf.math.scalar_mul(alpha, x),
                            tf.expand_dims(tf.math.reduce_max(
                                tf.math.scalar_mul(alpha, x), axis=1),
                                           axis=1)),
                        tf.expand_dims(tf.math.log(
                            tf.math.reduce_sum(tf.math.exp(
                                tf.subtract(
                                    tf.math.scalar_mul(alpha, x),
                                    tf.expand_dims(tf.math.reduce_max(
                                        tf.math.scalar_mul(alpha, x), axis=1),
                                                   axis=1))),
                                               axis=1)),
                                       axis=1))), w)

            npa = np.array(filt_list)
            print(npa, file=f)
            # print(npa.shape[0])
            for i in range(npa.shape[0]):
                df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T']).T
                # print(df.head())
                df.to_csv('kernel_weights/6.csv',
                          mode='a',
                          sep='\t',
                          float_format='%.3f')

            for i in range(npa.shape[0]):
                for rows in range(npa[i].shape[0]):
                    ownlog = np.array(npa[i][rows])
                    for cols in range(ownlog.shape[0]):
                        ownlog[cols] = ownlog[cols] * np.log2(ownlog[cols])
                    scalar = np.cumsum(ownlog, axis=0) + 2
                    npa[i][rows] *= scalar
                df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T'])
                print(df.head())
                logo = lm.Logo(
                    df,
                    font_name='Arial Rounded MT Bold',
                )
                # plt.show()
                plt.savefig('logos/l6/logo' + str(layer_num) + '_' + str(i) +
                            '.png',
                            dpi=50)


class Evaluation:
    # TODO: Move these methods to Prediction class in parent directory and
    # delete this module
    def __init__(self):
        self._model = self._load_model()

    def _load_model(self) -> keras.Model:
        # Find parent directory path dynamically
        parent_dir = Path(inspect.getabsfile(inspect.currentframe())).parent

        parameter_file = f'{parent_dir}/parameter1.txt'

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
        # tf.keras.utils.plot_model(
        #     model, to_file='model.png', show_shapes=False, show_dtype=False,
        #     show_layer_names=False, rankdir='TB', expand_nested=False, dpi=96
        # )

        print('Loading weights in model...')
        model.load_weights(f'{parent_dir}/model_weights/w6.h5_archived')
        return model

    def _plot_scatter(self, df: pd.DataFrame) -> None:
        p = (
            ggplot(data=df, mapping=aes(x='True Value', y='Predicted Value')) +
            stat_bin_2d(bins=150) + xlim(-2.75, 2.75) + ylim(-2.75, 2.75))

        with open(f'figures/scatter.png', 'w') as f:
            print(p, file=f)

    def check_performance(self, df: pd.DataFrame) -> None:
        """
        Check model performance on a sequence library and return predicted values.
        """
        start_time = time.time()

        prep = Preprocess(df)
        data = prep.one_hot_encode()

        x1 = data["forward"]
        x2 = data["reverse"]
        y = data["target"]

        history2 = self._model.evaluate({'forward': x1, 'reverse': x2}, y)

        print('metric values of model.evaluate: ' + str(history2))
        print('metrics names are ' + str(self._model.metrics_names))

        print(f"Took --- {time.time() - start_time} seconds ---")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        x1 = data["forward"]
        x2 = data["reverse"]

        y_pred = self._model.predict({'forward': x1, 'reverse': x2}).flatten()
        return df.assign(c0_predict=y_pred)

    def print_prediction_metrics(self, df: pd.DataFrame) -> None:
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        x1 = data["forward"]
        x2 = data["reverse"]
        y = data["target"]

        y_pred = self._model.predict({'forward': x1, 'reverse': x2}).flatten()
        assert y_pred.shape == y.shape
        print('r2 score:', r2_score(y, y_pred))
        print('Pearson\'s correlation:', pearsonr(y, y_pred)[0])
        print('Spearman\'s correlation: ', spearmanr(y, y_pred)[0])

        # self._plot_scatter(df)


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)
    df = DNASequenceReader().get_processed_data()[CNL]
    Evaluation().check_performance(df)
