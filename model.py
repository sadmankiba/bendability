from __future__ import annotations

from reader import DNASequenceReader
from shape import find_valid_cols, run_dna_shape_r_wrapper
from util import cut_sequence, find_occurence_individual

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import boruta
from imblearn.over_sampling import RandomOverSampler

import math
from pathlib import Path
import random
import time
from collections import Counter
from typing import Union

def encode_shape(shape_arr: np.ndarray, shape_str: str, n_split: int) -> np.ndarray:
    """
    Encodes a 2D shape array into integers between 0 to (n_split - 1)

    Returns:
        A numpy 2D array of integers
    """
    saved_shape_values_file = Path(f'data/generated_data/{shape_str}_possib_values.tsv')
    if saved_shape_values_file.is_file():
        with open(saved_shape_values_file, 'r') as f:
            possib_shape_df = pd.read_csv(f, sep='\t')
            shape_min = min(possib_shape_df[shape_str])
            shape_max = max(possib_shape_df[shape_str])
    else:
        shape_min = shape_arr.min()
        shape_max = shape_arr.max()

    shape_range = ((shape_max - shape_min) / n_split) + 0.001
    return np.floor((shape_arr - shape_min) / shape_range)


def encode_shape_alpha(shape_arr: np.ndarray, shape_str: str, n_letters: int) -> np.ndarray:
    """
    Encodes a 2D shape array into 1D array of strings.

    The string contains letters from first n_letters letters, ['a', ... ]

    Returns:
        A numpy 1D array of binary strings
    """
    enc_shape_arr = encode_shape(shape_arr, shape_str, n_letters)
    enc_shape_arr += ord('a')
    return enc_shape_arr.astype(np.uint8).view(f'S{shape_arr.shape[1]}').squeeze()


def one_hot_encode_shape(shape_arr, shape_str, n_split):
    """
    Transform shape values to discrete numbers and perform One-hot encoding

    args:
        shape_arr (numpy.Ndarray): shape values of DNA sequences
        shape_str: a str from ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
        n_split: number of discrete values

    returns:
        a numpy 3D array containing one-hot encoded shape values of sequences
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
        1D arr classified into integer between 0 to range_split.size - 1
    """
    range_arr = np.array(sorted(arr))[np.floor(arr.size * range_split - 1).astype(int)]     # ranges for array according to range_split

    return np.array([ np.searchsorted(range_arr, e) for e in arr ])


def get_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select rows with first and last class

    Args:
        df: A DataFrame in which `C0` column contains classes as denoted by integer from 0 to n-1
    """
    n = df['C0'].unique().size
    df = df.loc[((df['C0'] == 0) | (df['C0'] == n - 1))]
    df['C0'] = df['C0'] // (n - 1)

    return df


def get_balanced_classes(X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) \
    -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
    """
    Balance data according to classes
    """
    print('Before oversampling:', sorted(Counter(y).items()))
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print('After oversampling:', sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled


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

    def _prepare_shape_normalized(self, df: pd.DataFrame, shape_name: str) -> np.ndarray:
        """
        Get shape array of DNA sequences for training model

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with normalized shape values.
        """
        shape_arr = run_dna_shape_r_wrapper(df, True)[shape_name]
        print('shape_arr', shape_arr.shape)

        # Normalize
        shape_arr = (shape_arr - shape_arr.min()) / (shape_arr.max() - shape_arr.min())

        return shape_arr.reshape(shape_arr.shape[0], 1, shape_arr.shape[1])


    def _prepare_shape_encoded(self, df: pd.DataFrame, shape_name: str) -> np.ndarray:
        """
        Get shape array of DNA sequences

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with one-hot encoded shape values.
        """
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
        return X


    def _train_shape_cnn_classifier(self, X: np.ndarray, y: np.ndarray) \
        -> tuple[models.Sequential, tf.keras.callbacks.History]:
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
        kernel_size = 8
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(X.shape[1], kernel_size), strides=1,
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

        history = model.fit(X_train, y_train, epochs=15,
                            validation_data=(X_valid, y_valid))

        return model, history


    def run_shape_cnn_classifier(self, shape_name: str, c0_range_split: np.ndarray, \
            encode: bool, binary: bool) -> None:
        """
        Run a CNN classifier on DNA shape values.

        Args:
            shape_name: Name of structural feature
            c0_range_split: A numpy 1D array denoting the point of split for classification
            encode: whether to encode with one-hot-encoding
        """

        df = self._get_train_df()

        # Classify
        df['C0'] = classify_arr(df['C0'].to_numpy(), c0_range_split)
        if binary:
            df = get_binary_classification(df)
        y = df['C0'].to_numpy()

        # Prepare X
        if encode:
            X = self._prepare_shape_encoded(df, shape_name)
        else:
            X = self._prepare_shape_normalized(df, shape_name)

        X = X.reshape(list(X.shape) + [1])



        # Balance classes
        # X, y = get_balanced_classes(X, y)

        # Train
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.1)
        model, history = self._train_shape_cnn_classifier(X_train_valid, y_train_valid)
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
        print('test_acc', test_acc)

        # Plot
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()


    def _select_feat(self, X: np.ndarray, y: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Select all relevant features with Boruta algorithm

        Args:
            X: feature array
            y: target

        Returns:
            A tuple containing
                * A numpy ndarray: X containing selected features
                * A numpy 1D array of boolean denoting which features were selected
                * A numpy 1D array of integer denoting ranking of features

        """
        t = time.time()
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        feat_selector = boruta.BorutaPy(rf, n_estimators='auto', verbose=1, \
            random_state=1, perc=90, max_iter=50)
        feat_selector.fit(X, y)

        print(f'Boruta run time: {(time.time() - t) / 60} min')

        return feat_selector.transform(X), feat_selector.support_, feat_selector.ranking_


    def run_seq_classifier(self, k_list: list[int], c0_range_split: np.ndarray, binary: bool) -> None:
        """
        Runs Scikit-learn classifier to classify C0 value with k-mer count.

        Prepares features and targets from DNA sequences and C0 value.

        Args:
            k_list: list of unit sizes to consider
            c0_range_split: A numpy 1D array denoting the point of split for classification
            binary: whether to perform binary classification
        """
        # Get count features
        df = self._get_train_df()
        t = time.time()
        df = find_occurence_individual(df, k_list)
        print(f'Substring count time: {(time.time() - t) / 60} min')

        # Save for inspection
        k_list_str = ''.join([ str(k) for k in k_list ])
        file_name = f'{self.library_name}_{self.seq_start_pos}_{self.seq_end_pos}_kmercount_{k_list_str}.tsv'
        df.sort_values('C0').to_csv(f'data/generated_data/kmer_count/{file_name}', sep='\t', index=False)

        # Classify
        df['C0'] = classify_arr(df['C0'].to_numpy(), c0_range_split)

        if binary:
            df = get_binary_classification(df)

        # Balance classes
        df, _ = get_balanced_classes(df, df['C0'].to_numpy())

        # Shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        y = df['C0'].to_numpy()

        # Normalize features
        df = df.drop(columns=['Sequence #', 'Sequence', 'C0'])
        df = (df - df.mean()) / df.std()

        # Select features with Boruta algorithm
        cols = df.columns.to_numpy()
        X = df.to_numpy()
        print('X', X.shape)
        print('5 random rows of features')
        print(X[random.sample(range(X.shape[0]), 5)])
        print('5 random targets')
        print(y[random.sample(range(X.shape[0]), 5)])
        X_sel, sel_cols, sel_rank = self._select_feat(X, y)
        print('After feature selection, X_sel', X_sel.shape)

        # Train - 3 fold cross validation
        skf = StratifiedKFold(n_splits = 3)
        for idx, (train_idx, test_idx) in enumerate(skf.split(X_sel, y)):
            print('In fold:', idx + 1)

            X_train = X_sel[train_idx]
            y_train = y[train_idx]

            X_test = X_sel[test_idx]
            y_test = y[test_idx]
            forest = RandomForestClassifier(n_estimators=5, random_state=2)
            forest.fit(X_train, y_train)

            print('accuracy:', forest.score(X_test, y_test))


    def run_shape_seq_classifier(self) -> None:
        pass


if __name__ == '__main__':
    model = Model('cnl', 1, 50)
    shape_name = 'ProT'
    k_list = [2, 3, 4, 5]
    c0_range_split = np.array([0.25, 0.75, 1.0])
    model.run_seq_classifier(k_list, c0_range_split, binary=True)
    # model.run_shape_cnn_classifier(shape_name, c0_range_split, encode=False, binary=True)
