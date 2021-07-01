from __future__ import annotations

from matplotlib.pyplot import fill

from util import cut_sequence, find_occurence_individual, find_helical_separation
from shape import run_dna_shape_r_wrapper
from reader import DNASequenceReader

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import boruta

import math 
from collections import Counter
from typing import Union, TypedDict
from pathlib import Path
import time
import random

class SequenceLibrary(TypedDict):
    """
    Attributes:
        name: Name of the library to use for training / testing
        seq_start_pos: Start position to consider (between 1-50)
        seq_end_pos: End position to consider (between 1-50)
    """
    name: str
    seq_start_pos: int
    seq_end_pos: int


class DataOrganizeOptions(TypedDict):
    """
    Attributes:
        k_list: list of unit sizes to consider when using k-mers
        range_split: Range list for classification of targets. A numpy
            array of float values between 0 and 1 that sums to 1. eg. [0.3, 0.5, 0.2]
        binary_class: Whether to perform binary classification
        balance: Whether to balance classes
    """
    k_list: list[int]
    range_split: np.ndarray 
    binary_class: bool
    balance: bool


class ShapeOrganizer:
    """
    Encodes DNA shape values

    Attributes:
        shape_str: Name of the shape. Valid valued: 'HelT', 'ProT', 'Roll', 'MGW', 'EP'
    """
    def __init__(self, shape_str: str, library: SequenceLibrary):
        self.shape_str = shape_str
        self._library = library


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


    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        return run_dna_shape_r_wrapper(df, True)[self.shape_str]


class AlphaShapeEncoder(ShapeOrganizer):
    """
    Encodes DNA shape values into alphabetical letters
    """
    def __init__(self, shape_str: str, library: SequenceLibrary):
        super().__init__(shape_str, library)
        

    def _encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
        """
        Encodes a 2D shape array into 1D array of strings.

        The string contains letters from first n_letters letters, ['a', ... ]

        Returns:
            A numpy 1D array of binary strings
        """
        enc_shape_arr = self._encode_shape_n(shape_arr, n_split)
        enc_shape_arr += ord('a')
        return enc_shape_arr.astype(np.uint8).view(f'S{shape_arr.shape[1]}').squeeze()


    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        pass
        

class OheShapeEncoder(ShapeOrganizer):
    """
    Encodes DNA shape values with one-hot encoding
    """
    def __init__(self, shape_str: str, library: SequenceLibrary):
        super().__init__(shape_str, library)

    
    def _encode_shape(self, shape_arr: np.ndarray, n_split: int) -> np.ndarray:
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


    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get shape array of DNA sequences

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with one-hot encoded shape values.
        """
        shape_arr = super().prepare_shape(df)
        print('shape_arr', shape_arr.shape)

        num_shape_encode = 12 # CHANGE HERE

        saved_shape_arr_file = Path(f"data/generated_data/saved_arrays/{self._library['name']}_{self._library['seq_start_pos']}_{self._library['seq_end_pos']}_{self.shape_str}_ohe.npy")
        if saved_shape_arr_file.is_file():
            # file exists
            with open(saved_shape_arr_file, 'rb') as f:
                X = np.load(f)
        else:
            X = self._encode_shape(shape_arr, num_shape_encode)
            with open(saved_shape_arr_file, 'wb') as f:
                np.save(f, X)

        print('X.shape', X.shape)
        print(X[random.sample(range(X.shape[0]), 5)])
        assert X.shape == (shape_arr.shape[0], num_shape_encode, shape_arr.shape[1])
        return X


class ShapeNormalizer(ShapeOrganizer):
    """
    Normalizes DNA shape values into alphabetical letters
    """
    def __init__(self, shape_str: str, library):
        super().__init__(shape_str, library)
    

    def prepare_shape(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get shape array of DNA sequences for training model

        Args:
            df: A Dataframe
            shape_name: Name of the structural feature

        Returns:
            A numpy 3D array with normalized shape values.
        """
        shape_arr = super().prepare_shape(df)
        print('shape_arr', shape_arr.shape)

        # Normalize
        shape_arr = (shape_arr - shape_arr.min()) / (shape_arr.max() - shape_arr.min())

        return shape_arr.reshape(shape_arr.shape[0], 1, shape_arr.shape[1])


class ShapeOrganizerFactory:
    """
    Generates shape organizer object
    """
    def __init__(self, organizer_type: str, shape_str: str):
        """
        Constructor
        
        Args:
            organizer_type: Valid values- ['alpha', 'ohe', 'normal'].
            shape_str: Valid values- ['HelT', 'ProT', 'Roll', 'MGW', 'EP']. 
        """
        self._organizer_type = organizer_type
        self._shape_str = shape_str


    def make_shape_organizer(self, library: SequenceLibrary) -> ShapeOrganizer:
        """
        Creates shape encoder object according to encode type
            
        Returns:
            Created shape encoder object
        """
        if self._organizer_type == 'alpha':
            return AlphaShapeEncoder(self._shape_str, library)
        elif self._organizer_type == 'ohe':
            return OheShapeEncoder(self._shape_str, library)
        elif self._organizer_type == 'normal':
            return ShapeNormalizer(self._shape_str, library)
        else:
            raise ValueError('Organizer type not recognized')


class FeatureSelector:
    """
    Selects Features for classification

    Attributes
        support_: A numpy 1D array of boolean denoting which features were selected
        ranking: A numpy 1D array of integer denoting ranking of features
    """
    def __init__(self):
        """
        Selects features
        """
        self.support_ = None 
        self.ranking_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception('Subclass responsibility')
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise Exception('Subclass responsibility')


class ManualFeatureSelector(FeatureSelector):
    """
    Select features that are strictly increasing/decreasing for classes 0 to
    n-1.
    """
    def _check_increasing(self, arr: np.ndarray[float]) -> np.ndarray[bool]:
        return np.all(arr[1:, :] > arr[:-1, :], axis=0)


    def _check_decreasing(self, arr: np.ndarray[float]) -> np.ndarray[bool]:
        return np.all(arr[1:, :] < arr[:-1, :], axis=0)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        mean_list = []
        for i in np.unique(y):
            mean_list.append(X[np.where(y == i)].mean(axis=0))

        mean_arr = np.array(mean_list)
        assert mean_arr.shape == (np.unique(y).size, X.shape[1])

        # Check which columns are either increasing or decreasing 
        self.support_ = self._check_increasing(mean_arr) | self._check_decreasing(mean_arr)
        self.ranking_ = (~self.support_).astype(int) + 1
    
            
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


class BorutaFeatureSelector(FeatureSelector):
    """
    A wrapper class for Boruta feature selection algorithm
    """

    def __init__(self):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        
        self._feat_selector = boruta.BorutaPy(rf, n_estimators='auto', verbose=0, \
            random_state=1, perc=90, max_iter=50)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Select all relevant features with Boruta algorithm

        Args:
            X: feature array
            y: target

        Returns:
            None
        """
        self._feat_selector.fit(X, y)
        
        self.support_ = self._feat_selector.support_
        self.ranking_ = self._feat_selector.ranking_


    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._feat_selector.transform(X)


class AllFeatureSelector(FeatureSelector):
    """
    Selects all features
    """
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.support_ = np.full(shape=(X.shape[1],), fill_value=True)
        self.ranking_ = np.full(shape=(X.shape[1],), fill_value=1)
        

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


class FeatureSelectorFactory:
    def __init__(self, selector_type):
        self.selector_type = selector_type

    def make_feature_selector(self):
        '''Creates feature selector instance'''
        if self.selector_type == 'manual':
            return ManualFeatureSelector()
        elif self.selector_type == 'boruta':
            return BorutaFeatureSelector()
        elif self.selector_type == 'all':
            return AllFeatureSelector()
        else:
            raise Exception('Selector type not recognized')


class ClassificationMaker:
    '''Helper functions for classification'''
    def __init__(self, range_split: np.ndarray, binary_class: bool):
        """
        Creates a ClassificationMaker object. 
        """
        self._range_split = range_split
        self._binary_class = binary_class


    def _classify_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Encodes a 1D numpy array into discrete values

        Args:
            arr: 1D numpy array

        Returns:
            1D arr classified into integer between 0 to self._options['range_split'].size - 1
        """
        accumulated_range = np.cumsum(self._range_split)
        assert math.isclose(accumulated_range[-1], 1.0, rel_tol=1e-4)

        # Find border values for ranges
        range_arr = np.array(sorted(arr))[np.floor(arr.size * accumulated_range - 1).astype(int)]    

        return np.array([ np.searchsorted(range_arr, e) for e in arr ])


    def _get_binary_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select rows with first and last class

        Args:
            df: A DataFrame in which `C0` column contains classes as denoted by integer from 0 to n-1
        
        Returns:
            A dataframe with selected rows
        """
        df = df.copy()
        # Select rows that contain only first and last class
        n = df['C0'].unique().size
        df = df.loc[((df['C0'] == 0) | (df['C0'] == n - 1))]

        # Change last class's value to 1
        df['C0'] = df['C0'].apply(lambda val: val // (n - 1))

        return df
    

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df['C0'] = self._classify_arr(df['C0'].to_numpy())

        if self._binary_class:
            df = self._get_binary_classification(df)

        return df 


    def get_balanced_classes(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) \
        -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """
        Balance data according to classes
        """
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        return X_resampled, y_resampled


class DataOrganizer:
    """
    Prepares features and targets.
    """
    def __init__(self, library: SequenceLibrary, shape_organizer: ShapeOrganizer, 
            selector: FeatureSelector, options: DataOrganizeOptions):
        """
        Creates a DataOrganizer object.

        Args:
            library: SequenceLibrary object
            shape_organizer: Shape organizer object
            selector: Feature selector
            options: Options to prepare feature and target
        """
        self._library = library
        self._shape_organizer = shape_organizer
        self._feat_selector = selector
        self._options = options
        self._df = None
        self._class_maker = ClassificationMaker(
            self._options['range_split'], self._options['binary_class'])        


    def _prepare_data(self):
        reader = DNASequenceReader()
        all_df = reader.get_processed_data()

        return cut_sequence(all_df[self._library['name']], self._library['seq_start_pos'], self._library['seq_end_pos'])


    def _get_data(self):
        """
        If data not read, reads data and returns it.
        
        Data is not read in constructor to speed up tests which create DataOrganizer objects.
        """
        if self._df is not None:
            return self._df
        
        self._df = self._prepare_data()
        return self._df


    def _save_classification_data(self, df: pd.DataFrame) -> None:
        '''Save classification data in a tsv file for inspection'''
        k_list_str = ''.join([ str(k) for k in self._options['k_list'] ])
        classify_str = '_'.join([str(int(val * 100)) for val in self._options['range_split']])
        file_name = f'{self._library["name"]}_{self._library["seq_start_pos"]}_{self._library["seq_end_pos"]}_kmercount_{k_list_str}_{classify_str}'
        
        df.sort_values('C0').to_csv(f'data/generated_data/classification/{file_name}.tsv', sep='\t', index=False)
        df = df.drop(columns=['Sequence #', 'Sequence'])
        df.groupby('C0').mean().sort_values('C0')\
            .to_csv(f'data/generated_data/kmer_count/{file_name}_mean.tsv', sep='\t', index=False)


    def _get_helical_sep(self) -> pd.DataFrame:
        """Count helical separation. Load if already saved."""
        
        saved_helical_sep_file = Path(f"data/generated_data/helical_separation"
            f"/{self._library['name']}_{self._library['seq_start_pos']}_{self._library['seq_end_pos']}_hs.tsv")
        
        if saved_helical_sep_file.is_file():
            # file exists
            with open(saved_helical_sep_file, 'r') as f:
                df_hel = pd.read_csv(f, sep='\t')
                return df_hel
        
        t = time.time()
        df_hel = find_helical_separation(self._get_data())
        print(f'Helical separation count time: {(time.time() - t) / 60} min')
        
        df_hel.to_csv(saved_helical_sep_file, sep='\t', index=False)
        
        return df_hel


    def _get_kmer_count(self) -> pd.DataFrame:
        '''Get k-mer count features. Load if already saved.'''
        df = self._get_data()
        df_kmer = df

        for k in self._options['k_list']: 
            saved_kmer_count_file = Path(f"data/generated_data/kmer_count"
                f"/{self._library['name']}_{self._library['seq_start_pos']}"
                f"_{self._library['seq_end_pos']}_kmercount_{k}.tsv")
            
            if saved_kmer_count_file.is_file():
                # file exists
                with open(saved_kmer_count_file, 'r') as f:
                    df_one_kmer = pd.read_csv(f, sep='\t')
            else:            
                t = time.time()
                df_one_kmer = find_occurence_individual(df, [k])
                print(f'{k}-mer count time: {(time.time() - t) / 60} min')
                df_one_kmer.to_csv(saved_kmer_count_file, sep='\t', index=False)
                
            df_kmer = df_kmer.merge(df_one_kmer, on=['Sequence #', 'Sequence', 'C0'])
            
        return df_kmer


    def get_seq_train_test(self, classify: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares features and targets from DNA sequences and C0 value.
        """       
        df_kmer = self._get_kmer_count()
        df_hel = self._get_helical_sep()
        
        df = df_kmer.merge(df_hel, on=['Sequence #', 'Sequence', 'C0'])
        print(df)

        if classify:
            df = self._class_maker.classify(df) 
            self._save_classification_data(df)

        # Split train-test data
        y = df['C0'].to_numpy()
        df = df.drop(columns=['Sequence #', 'Sequence', 'C0'])
        df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
        
        # Print sample train values
        X_train = df_train.to_numpy()
        X_test = df_test.to_numpy()
        print('X', X_train.shape)
        print('5 random rows of features')
        print(X_train[random.sample(range(X_train.shape[0]), 5)])
        print('5 random targets')
        print(y_train[random.sample(range(X_train.shape[0]), 5)])
        
        # Select features
        self._feat_selector.fit(X_train, y_train)
        X_train = self._feat_selector.transform(X_train)
        X_test = self._feat_selector.transform(X_test)
        print('After feature selection, X_sel', X_train.shape)
        # print('Selected features:', df_train.columns[self._feat_selector.support_])

        # Balance classes
        if classify and self._options['balance']:
            print('Before oversampling:', sorted(Counter(y_train).items()))
            X_train, y_train = self._get_balanced_classes(X_train, y_train)
            print('After oversampling:', sorted(Counter(y_train).items()))

        # Normalize features
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
        
        return X_train, X_test, y_train, y_test 


    def get_shape_train_test(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares features and targets from DNA shape values and C0 value.
        """
        df = self._get_data()()

        # Classify
        df = self._class_maker.classify(df)
        y = df['C0'].to_numpy()

        X = self._shape_organizer.prepare_shape(df)
        X = X.reshape(list(X.shape) + [1])

        # Balance classes
        # X, y = get_balanced_classes(X, y)

        return train_test_split(X, y, test_size=0.1)