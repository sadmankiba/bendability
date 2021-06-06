from contextlib import suppress
from os import stat
from deeper_model import nn_model
import sys
import time
import random
from data_preprocess2 import preprocess
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logomaker as lm
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import StandardScaler

from plotnine import ggplot, aes, xlim, ylim, stat_bin_2d

# get dictionary from text file


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


def main(argv=None):
    if argv is None:
        argv = sys.argv
        # input argszw
        file = argv[1]
        # e.g. data/41586_2020_3052_MOESM4_ESM.txt
        parameter_file = argv[2]
        #e.g. parameter1.txt

    # excute the code
    start_time = time.time()

    params = get_parameters(parameter_file)

    dim_num = (-1, 50, 4)
    nn = nn_model(dim_num=dim_num, filters=params["filters"], kernel_size=params["kernel_size"], pool_type=params["pool_type"], regularizer=params["regularizer"],
                  activation_type=params["activation_type"], epochs=params["epochs"], batch_size=params["batch_size"], loss_func=params["loss_func"], optimizer=params["optimizer"])
    model = nn.create_model()

    model.load_weights('model_weights_8.h5')

    np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)
    prep = preprocess(file)
    # if want mono-nucleotide sequences
    data = prep.one_hot_encode()
    # if want dinucleotide sequences
    #dict = prep.dinucleotide_encode()

    readout = data["readout"]
    fw_fasta = data["forward"]
    rc_fasta = data["reverse"]

    # if params["activation_type"] == 'linear':
    #     readout = np.log2(readout)
    #     readout = np.ndarray.tolist(readout)

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
    
    x1 = np.asarray(fw_fasta)
    x2 = np.asarray(rc_fasta)
    y = np.asarray(readout)

    history2 = model.evaluate(
        {'forward': x1, 'reverse': x2}, y)

    print('metric values of model.evaluate: ' + str(history2))
    print('metrics names are ' + str(model.metrics_names))

    predictions = model.predict(
        {'forward': x1, 'reverse': x2}).flatten()
    folder = '.'

    df = pd.DataFrame(
        {'sequences': prep.read_fasta_forward(), 'predicted_value': predictions, 'true_value': y})
    df.to_csv(folder + '/predictions_8_9.csv',
              sep='\t', index=False, header=False)
    # df.plot('x', 'y', kind='scatter')
    # plt.show()

    p = (ggplot(data=df,
                mapping=aes(x='true_value', y='predicted_value'))
         + stat_bin_2d(bins=150)
         + xlim(-2.75, 2.75)
         + ylim(-2.75, 2.75)
         )
    print(p)

    # history2 = model.evaluate({'forward': x1_test, 'reverse': x2_test}, y1_test)
    # pred = model.predict({'forward': x1_test, 'reverse': x2_test})

    # #viz_prediction(pred, y1_test, '{} regression model'.format(self.loss_func), '{}2.png'.format(self.loss_func))

    # folder = '.'
    # with open(folder+'/weights2', 'w') as f:
    #     for layer_num in range(2, 6):
    #         layer = model.layers[layer_num]
    #         config = layer.get_config()
    #         print(config, file=f)
    #         weights = layer.get_weights()
    #         w = tf.transpose(weights[0], [2, 0, 1])
    #         alpha = 20
    #         beta = 1/alpha
    #         bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
    #         bkg_tf = tf.cast(bkg, tf.float32)
    #         filt_list = tf.map_fn(lambda x:
    #                             tf.math.exp(
    #                                 tf.subtract(
    #                                     tf.subtract(
    #                                         tf.math.scalar_mul(alpha, x),
    #                                         tf.expand_dims(
    #                                             tf.math.reduce_max(
    #                                                 tf.math.scalar_mul(alpha, x),
    #                                                 axis=1
    #                                             ),
    #                                             axis=1
    #                                         )
    #                                     ),
    #                                     tf.expand_dims(
    #                                         tf.math.log(
    #                                             tf.math.reduce_sum(
    #                                                 tf.math.exp(
    #                                                     tf.subtract(
    #                                                         tf.math.scalar_mul(
    #                                                             alpha, x),
    #                                                         tf.expand_dims(
    #                                                             tf.math.reduce_max(
    #                                                                 tf.math.scalar_mul(
    #                                                                     alpha, x),
    #                                                                 axis=1
    #                                                             ),
    #                                                             axis=1
    #                                                         )
    #                                                     )
    #                                                 ),
    #                                                 axis=1
    #                                             )
    #                                         ),
    #                                         axis=1
    #                                     )
    #                                 )
    #                             ), w)

    #         npa = np.array(filt_list)
    #         print(npa, file=f)
    #         print(npa.shape[0])
            
    #         for i in range(npa.shape[0]):
    #             for rows in range(npa[i].shape[0]):
    #                 ownlog = np.array(npa[i][rows])
    #                 for cols in range(ownlog.shape[0]):
    #                     ownlog[cols] = ownlog[cols] * np.log2(ownlog[cols])
    #                 scalar = np.cumsum(ownlog, axis=0) + 2
    #                 npa[i][rows] *= scalar
    #             df = pd.DataFrame(npa[i], columns=['A', 'C', 'G', 'T'])
    #             print(df.head())
    #             logo = lm.Logo(df, font_name='Arial Rounded MT Bold', )  
    #             # plt.show()
    #             plt.savefig('logos2/logo' + str(layer_num) + '_' + str(i) + '.png', dpi=50)
        

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    sys.exit(main())
