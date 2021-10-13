import time
import sys
import random
import inspect
import keras

import numpy as np
import tensorflow as tf

from .data_preprocess import Preprocess
from .model6 import nn_model
from .prediction import get_parameters 
from util.constants import TL, RL
from util.reader import DNASequenceReader
from util.util import PathUtil


def train(save=False) -> keras.Model:
    # TODO: Use argparse to overwrite parameter config
    train_lib = TL
    val_lib = RL

    # excute the code
    start_time = time.time()
    # Reproducibility
    seed = random.randint(1, 1000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    params = get_parameters(f'{PathUtil.get_parent_dir(inspect.currentframe())}/parameter_model6.txt')

    dim_num = (-1, 50, 4)
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
    model = nn.create_model()

    train_prep = Preprocess(DNASequenceReader().get_processed_data()[train_lib][:1000])
    # if want mono-nucleotide sequences
    train_data = train_prep.one_hot_encode()
    # if want dinucleotide sequences
    #dict = prep.dinucleotide_encode()

    val_prep = Preprocess(DNASequenceReader().get_processed_data()[val_lib][:1000])
    # if want mono-nucleotide sequences
    val_data = val_prep.one_hot_encode()
    # if want dinucleotide sequences
    #dict = prep.dinucleotide_encode()

    np.set_printoptions(threshold=sys.maxsize)

    # change from list to numpy array
    y_train = train_data["target"]
    x1_train = train_data["forward"]
    x2_train = train_data["reverse"]

    y_val = val_data["target"]
    x1_val = val_data["forward"]
    x2_val = val_data["reverse"]

    # Without early stopping
    model.fit({'forward': x1_train, 'reverse': x2_train}, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_data=({
                'forward': x1_val,
                'reverse': x2_val
            }, y_val))

    # Early stopping
    #callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    #callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    #history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks = [callback])

    print("Seed number is {}".format(seed))

    if save:
        model.save_weights('model_weights/w6.h5', save_format='h5')

    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))
    return model
