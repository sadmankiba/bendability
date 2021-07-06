from constants import LIBRARIES, Library
from data_preprocess import Preprocess
from model6 import nn_model

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
from plotnine import ggplot, aes, xlim, ylim, stat_bin_2d

import shutil

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


def check_performance(model, library: Library):
    prep = Preprocess(f'data/{library["file"]}')
    # if want mono-nucleotide sequences
    data = prep.one_hot_encode()
    # if want dinucleotide sequences
    #dict = prep.dinucleotide_encode()

    target = data["target"]
    fw = data["forward"]
    rc = data["reverse"]

    # scaler = StandardScaler()
    # reshaped_readout = np.asarray(readout).reshape(len(readout), 1)
    # scaler.fit(reshaped_readout)
    # norm_readout = scaler.transform(reshaped_readout).flatten()

    # 90% Train, 10% Test
    # x1_train, x1_test, y1_train, y1_test = train_test_split(
    #     fw_fasta, readout, test_size=0.1, shuffle=False)
    # x2_train, x2_test, y2_train, y2_test = train_test_split(
    #     rc_fasta, readout, test_size=0.1, shuffle=False)

    # s_train, s_test, y_train, y_test = train_test_split(
    #     prep.read_fasta_forward(), readout, test_size=0.1, shuffle=False)

    # # change from list to numpy array
    # y1_train = np.asarray(y1_train)
    # y1_test = np.asarray(y1_test)
    # y2_train = np.asarray(y2_train)
    # y2_test = np.asarray(y2_test)
    
    x1 = np.asarray(fw)
    x2 = np.asarray(rc)
    y = np.asarray(target)

    history2 = model.evaluate({'forward': x1, 'reverse': x2}, y)

    print('metric values of model.evaluate: ' + str(history2))
    print('metrics names are ' + str(model.metrics_names))

    y_pred = model.predict({'forward': x1, 'reverse': x2}).flatten()
    assert y_pred.shape == y.shape
    print('r2 score:', r2_score(y, y_pred))

    df = pd.DataFrame(
        {'Sequence': prep.df['Sequence'].str[25:-25].tolist(), 'Predicted Value': y_pred, 'True Value': y})
    
    df.to_csv(f'predictions/{library["name"]}_pred.csv', sep='\t', index=False)
    print('Predictions saved.')
    
    # Plot scatter plot
    p = (ggplot(data=df,
                mapping=aes(x='True Value', y='Predicted Value'))
         + stat_bin_2d(bins=150)
         + xlim(-2.75, 2.75)
         + ylim(-2.75, 2.75)
         )

    with open(f'figures/scatter_{library["name"]}.png', 'w') as f:
        print(p, file=f)


def save_kernel_weights_logos(model):
    with open('kernel_weights/6', 'w') as f:
        for layer_num in range(2, 3):
            layer = model.layers[layer_num]
            config = layer.get_config()
            print(config, file=f)
            weights = layer.get_weights()
            w = tf.transpose(weights[0], [2, 0, 1])
            alpha = 20
            beta = 1/alpha
            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x:
                                tf.math.exp(
                                    tf.subtract(
                                        tf.subtract(
                                            tf.math.scalar_mul(alpha, x),
                                            tf.expand_dims(
                                                tf.math.reduce_max(
                                                    tf.math.scalar_mul(alpha, x),
                                                    axis=1
                                                ),
                                                axis=1
                                            )
                                        ),
                                        tf.expand_dims(
                                            tf.math.log(
                                                tf.math.reduce_sum(
                                                    tf.math.exp(
                                                        tf.subtract(
                                                            tf.math.scalar_mul(
                                                                alpha, x),
                                                            tf.expand_dims(
                                                                tf.math.reduce_max(
                                                                    tf.math.scalar_mul(
                                                                        alpha, x),
                                                                    axis=1
                                                                ),
                                                                axis=1
                                                            )
                                                        )
                                                    ),
                                                    axis=1
                                                )
                                            ),
                                            axis=1
                                        )
                                    )
                                ), w)

            npa = np.array(filt_list)
            print(npa, file=f)
            # print(npa.shape[0])
            for i in range(npa.shape[0]):
                df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T']).T
                # print(df.head())
                df.to_csv('kernel_weights/6.csv', mode='a', sep='\t', float_format='%.3f')
            
            for i in range(npa.shape[0]):
                for rows in range(npa[i].shape[0]):
                    ownlog = np.array(npa[i][rows])
                    for cols in range(ownlog.shape[0]):
                        ownlog[cols] = ownlog[cols] * np.log2(ownlog[cols])
                    scalar = np.cumsum(ownlog, axis=0) + 2
                    npa[i][rows] *= scalar
                df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T'])
                print(df.head())
                logo = lm.Logo(df, font_name='Arial Rounded MT Bold', )  
                # plt.show()
                plt.savefig('logos/l6/logo' + str(layer_num) + '_' + str(i) + '.png', dpi=50)
    

def main():
    argv = sys.argv
    parameter_file = argv[1] #e.g. parameter1.txt

    analyse_library = 'tl'
    
    start_time = time.time()

    params = get_parameters(parameter_file)

    dim_num = (-1, 50, 4)
    print('Initializing nn_model object...')
    nn = nn_model(dim_num=dim_num, filters=params["filters"], kernel_size=params["kernel_size"], pool_type=params["pool_type"], regularizer=params["regularizer"],
                  activation_type=params["activation_type"], epochs=params["epochs"], batch_size=params["batch_size"], loss_func=params["loss_func"], optimizer=params["optimizer"])
    
    print('Creating Keras model...')
    model = nn.create_model()
    # tf.keras.utils.plot_model(
    #     model, to_file='model.png', show_shapes=False, show_dtype=False,
    #     show_layer_names=False, rankdir='TB', expand_nested=False, dpi=96
    # )

    print('Loading weights in model...')
    model.load_weights('model_weights/w6.h5')

    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)
    check_performance(model, LIBRARIES[analyse_library])

    # save_kernel_weights_logos(model)

    print(f"Took --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    sys.exit(main())
