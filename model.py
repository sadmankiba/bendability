from reader import DNASequenceReader
from shape import find_valid_cols, run_dna_shape_r_wrapper
from util import cut_sequence

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import math
from pathlib import Path
import random

# Two types of shape feature processing 
#   1. Only discrete values
#   2. One-hot encode discrete values

def encode_shape(shape_arr, shape_str, n_split):
    possib_shape_df = pd.read_csv(f'data/generated_data/{shape_str}_possib_values.tsv', sep='\t')
    shape_min = min(possib_shape_df[shape_str])
    shape_max = max(possib_shape_df[shape_str])

    shape_range = ((shape_max - shape_min) / n_split) + 0.001
    return np.floor((shape_arr - shape_min) / shape_range) 


def one_hot_encode_shape(shape_arr, shape_str, n_split):
    """
    Transform shape values to discrete numbers and perform One-hot encoding

    args:
        shape_arr (numpy.Ndarray): shape values of DNA sequences
        shape_str: a str from ['EP', 'HelT', 'MGW', 'ProT', 'Roll'] 
        n_split: number of discrete values
    
    returns:
        a numpy Ndarray containing one-hot encoded shape values of sequences 
    """
    encoded_shape_arr = encode_shape(shape_arr, shape_str, n_split)
    ohe = OneHotEncoder()
    ohe.fit(np.array(list(range(n_split))).reshape(-1,1))

    arr_3d = None
    for i in range(encoded_shape_arr.shape[0]):
        seq_enc_shape_arr = ohe.transform(encoded_shape_arr[i].reshape(-1,1)).toarray().transpose()
        seq_enc_shape_arr = seq_enc_shape_arr.reshape(1, seq_enc_shape_arr.shape[0], seq_enc_shape_arr.shape[1])
        if i == 0:
            arr_3d = seq_enc_shape_arr
        else:
            arr_3d = np.concatenate((arr_3d, seq_enc_shape_arr), axis=0)

    return arr_3d


def classify_arr(arr: np.ndarray, range_split: np.ndarray) -> np.ndarray:
    """
    Encodes a 1D numpy array into discrete values 

    args:
        arr: numpy array
        range_split: a numpy array of elements from 0 to 1 that sums to 1. eg. [0.3, 0.8, 1.0]
    
    returns:
        arr classified into integer between 0 to range_split.size - 1 
    """
    range_arr = np.array(sorted(arr))[np.floor(arr.size * range_split - 1).astype(int)]     # ranges for array according to range_split

    return np.array([ np.searchsorted(range_arr, e) for e in arr ]) 
        

class Model:
    """
    A class to train model on DNA mechanics libraries. 
    """

    def __init__(self, library_name: str, seq_start_pos: int, seq_end_pos: int):
        """
        Constructor 

        Args:
            library_name: Name of the library to use for training / testing
            seq_start_pos: Start position to consider (between 1-50)
            seq_end_pos: End position to consider (between 1-50)
        """
        self.library_name = library_name
        self.seq_start_pos = seq_start_pos
        self.seq_end_pos = seq_end_pos


    def _get_train_df(self):
        reader = DNASequenceReader()
        all_df = reader.get_processed_data() 
        
        return cut_sequence(all_df[self.library_name], self.seq_start_pos, self.seq_end_pos)
         
        
    def _prepare_shape(self, df):
        shape_name = 'HelT'
        shape_arr = run_dna_shape_r_wrapper(df, True)[shape_name] 
        print('shape_arr', shape_arr.shape)

        num_shape_encode = 12 # CHANGE HERE

        saved_shape_arr_file = Path(f"data/generated_data/saved_arrays/{self.library_name}_{self.seq_start_pos}_{self.seq_end_pos}_{shape_name}_ohe.npy")
        if saved_shape_arr_file.is_file():
            # file exists
            with open(saved_shape_arr_file, 'rb') as f:
                X = np.load(f)
        else: 
            X = one_hot_encode_shape(shape_arr, shape_name, num_shape_encode)
            with open(saved_shape_arr_file, 'wb') as f:
                np.save(f, X)
            
        print('X.shape', X.shape)
        print(X[random.sample(range(X.shape[0]), 5)])
        assert X.shape == (shape_arr.shape[0], num_shape_encode, shape_arr.shape[1])


    def _train_shape_cnn(X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Train a CNN model on data

        Args:
            X: training and validation samples
            y: training and validation targets

        Returns:
            A tuple containing 
                * model: a tensorflow sequential model
                * history: history object of training
        """

        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(X.shape[1], 4), strides=1,
            activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(np.unique(y).size))

        print(model.summary())

        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        history = model.fit(X_train, y_train, epochs=5, 
                            validation_data=(X_valid, y_valid))

        return model, history


    def run(self):
        # Prepare X and y
        df = self._get_train_df()
        X = self._prepare_shape(df)
        c0_range_split = np.array([0.2, 0.8, 1.0])  
        y = classify_arr(df['C0'].to_numpy(), c0_range_split)
        X = X.reshape(list(X.shape) + [1])

        # Train
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15)
        model, history = self._train_shape_cnn(X_train_valid, y_train_valid)        
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
        print('test_acc', test_acc)

        # Plot
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    model = Model('tl', 11, 50)
    model.run()