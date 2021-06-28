from __future__ import annotations

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

import math 
from collections import Counter
from typing import Union
from pathlib import Path


class ShapeEncoder:
    """
    Encodes DNA shape values

    Attributes:
        shape_str: Name of the shape. Valid valued: 'HelT', 'ProT', 'Roll', 'MGW', 'EP'
    """
    def __init__(self, shape_str: str):
        self.shape_str = shape_str
        pass


    def _encode_shape_n(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Encodes a 2D shape array into integers between 0 to (n_split - 1)

        Returns:
            A numpy 2D array of integers
        """
        saved_shape_values_file = Path(f'data/generated_data/{self.shape_str}_possib_values.tsv')
        if saved_shape_values_file.is_file():
            with open(saved_shape_values_file, 'r') as f:
                possib_shape_df = pd.read_csv(f, sep='\t')
                shape_min = min(possib_shape_df[self.shape_str])
                shape_max = max(possib_shape_df[self.shape_str])
        else:
            shape_min = shape_arr.min()
            shape_max = shape_arr.max()

        shape_range = ((shape_max - shape_min) / n_split) + 0.001
        return np.floor((shape_arr - shape_min) / shape_range)


    def encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        raise Exception('Subclass responsibility.')


class AlphaShapeEncoder(ShapeEncoder):
    """
    Encodes DNA shape values into alphabetical letters
    """
    def __init__(self, shape_str: str):
        super().__init__(shape_str)
        

    def encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Encodes a 2D shape array into 1D array of strings.

        The string contains letters from first n_letters letters, ['a', ... ]

        Returns:
            A numpy 1D array of binary strings
        """
        enc_shape_arr = self._encode_shape_n(shape_arr, n_split)
        enc_shape_arr += ord('a')
        return enc_shape_arr.astype(np.uint8).view(f'S{shape_arr.shape[1]}').squeeze()


class OheShapeEncoder(ShapeEncoder):
    """
    Encodes DNA shape values with one-hot encoding
    """
    def __init__(self, shape_str):
        super().__init__(shape_str)

    
    def encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Transform shape values to discrete numbers and perform One-hot encoding

        args:
            shape_arr (numpy.Ndarray): shape values of DNA sequences
            shape_str: a str from ['EP', 'HelT', 'MGW', 'ProT', 'Roll']
            n_split: number of discrete values

        returns:
            a numpy 3D array containing one-hot encoded shape values of sequences
        """
        encoded_shape_arr = self._encode_shape_n(shape_arr, n_split)
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


class ShapeEncoderFactory:
    """
    Generates shape encoder object
    """
    def __init__(self):
        pass

    def make_shape_encoder(self, encoder_type: str, shape_str: str) -> ShapeEncoder:
        """
        Creates shape encoder object according to encode type

        Args:
            encoder_type: Valid values: 'alpha' or 'ohe'. 
            shape_str: Valid valued: 'HelT', 'ProT', 'Roll', 'MGW', 'EP'. 
        
        Returns:
            Created shape encoder object
        """
        if encoder_type == 'alpha':
            return AlphaShapeEncoder(shape_str)
        elif encoder_type == 'ohe':
            return OheShapeEncoder(shape_str)
        else:
            raise ValueError('Encoder type not recognized')


class DataOrganizer:
    """
    Prepares features and targets.

    Attributes: 
        range_split: Range list for classification of targets. A numpy
            array of float values between 0 and 1 that sums to 1. eg. [0.3, 0.5, 0.2]
        shape_encoder: Shape encoder object
        binary_class: Whether to perform binary classification

    """
    def __init__(self, range_split: np.ndarray, shape_encoder: ShapeEncoder, binary_class: bool):
        self.range_split = range_split
        self.shape_encoder = shape_encoder
        self.binary_class = binary_class


    def _classify_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Encodes a 1D numpy array into discrete values

        Args:
            arr: numpy array

        Returns:
            1D arr classified into integer between 0 to self.range_split.size - 1
        """
        accumulated_range = np.cumsum(self.range_split)
        assert math.isclose(accumulated_range[-1], 1.0, rel_tol=1e-4)

        # Find border values for ranges
        range_arr = np.array(sorted(arr))[np.floor(arr.size * accumulated_range - 1).astype(int)]    

        return np.array([ np.searchsorted(range_arr, e) for e in arr ])


    def _get_binary_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select rows with first and last class

        Args:
            df: A DataFrame in which `C0` column contains classes as denoted by integer from 0 to n-1
        """
        # Select rows that contain only first and last class
        n = df['C0'].unique().size
        df = df.loc[((df['C0'] == 0) | (df['C0'] == n - 1))]

        # Change last class's value to 1
        df['C0'] = df['C0'] // (n - 1)

        return df
    

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df['C0'] = self._classify_arr(df['C0'].to_numpy())

        if self.binary_class:
            df = self._get_binary_classification(df)

        return df 


    def get_balanced_classes(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) \
        -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """
        Balance data according to classes
        """
        print('Before oversampling:', sorted(Counter(y).items()))
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        print('After oversampling:', sorted(Counter(y_resampled).items()))
        return X_resampled, y_resampled


    def get_encoded_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        return self.shape_encoder.encode_shape(shape_arr, n_split)