from __future__ import annotations
import math
import random
import string
import re as bre  # built-in re
import itertools as it
from pathlib import Path
import logging
import inspect 
from types import FrameType
from typing import Union, Any

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from nptyping import NDArray

from .custom_types import YeastChrNum
from .constants import SEQ_LEN

logging.basicConfig(level=logging.INFO)

def reverse_compliment_of(seq: str):
    # Define replacements
    rep = {"A": "T", "T": "A", 'G': 'C', 'C': 'G'}

    # Create regex pattern
    rep = dict((bre.escape(k), v) for k, v in rep.items())
    pattern = bre.compile("|".join(rep.keys()))

    # Replace and return reverse sequence
    return (pattern.sub(lambda m: rep[bre.escape(m.group(0))], seq))[::-1]


def append_reverse_compliment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends reverse compliment sequences to a dataframe
    """
    rdf = df.copy()
    rdf['Sequence'] = df['Sequence'].apply(
        lambda seq: reverse_compliment_of(seq))
    return pd.concat([df, rdf], ignore_index=True)


def sorted_split(df: pd.DataFrame,
                 n=1000,
                 n_bins=1,
                 ascending=False) -> list[pd.DataFrame]:
    """
    Sort data according to C0 value
    params:
        df: dataframe
        n: Top n data to use
        n_bins: Number of bins(dataframes) to split to
        ascending: sort order
    returns:
        A list of dataframes.
    """
    sorted_df = df.sort_values(by=['C0'], ascending=ascending)[0:n]

    return [
        sorted_df.iloc[start_pos:start_pos + math.ceil(n / n_bins)]
        for start_pos in range(0, n, math.ceil(n / n_bins))
    ]


def cut_sequence(df, start, stop):
    """
    Cut a sequence from start to stop position.
    start, stop - 1-indexed, inclusive
    """
    df['Sequence'] = df['Sequence'].str[start - 1:stop]
    return df


def get_possible_seq(size):
    """
    Generates all possible nucleotide sequences of particular length

    Returns:
        A list of sequences
    """

    possib_seq = ['']

    for _ in range(size):
        possib_seq = [
            seq + nc for seq in possib_seq for nc in ['A', 'C', 'G', 'T']
        ]

    return possib_seq


def get_possible_shape_seq(size: int, n_letters: int):
    """
    Generates all possible strings of particular length from an alphabet

    Args:
        size: Size of string
        n_letters: Number of letters to use

    Returns:
        A list of strings
    """
    possib_seq = ['']
    alphabet = [chr(ord('a') + i) for i in range(n_letters)]

    for _ in range(size):
        possib_seq = [seq + c for seq in possib_seq for c in alphabet]

    return possib_seq


def find_shape_occurence_individual(df: pd.DataFrame, k_list: list,
                                    n_letters: int):
    """
    Find occurences of all possible shape sequences for individual DNA sequences.

    Args:
        df: column `Sequence` contains DNA shape sequences
        k_list: list of unit sizes to consider
        n_letters: number of letters used to encode shape

    Returns:
        A dataframe with columns added for all considered unit nucleotide sequences.
    """
    possib_seq = []
    for k in k_list:
        possib_seq += get_possible_shape_seq(k, n_letters)

    for seq in possib_seq:
        df = df.assign(new_column=lambda x: x['Sequence'].str.count(seq))
        df = df.rename(columns={'new_column': seq})

    return df


def gen_random_sequences(n: int):
    """Generates n 50 bp random DNA sequences"""
    seq_list = []
    for _ in range(n):
        seq = ''

        for _ in range(50):
            d = random.random()
            if d < 0.25:
                c = 'A'
            elif d < 0.5:
                c = 'T'
            elif d < 0.75:
                c = 'G'
            else:
                c = 'C'

            seq += c

        seq_list.append(seq)

    return seq_list


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def roman_to_num(chr_num: YeastChrNum) -> int:
    rom_num_map = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VIII': 8,
        'IX': 9,
        'X': 10,
        'XI': 11,
        'XII': 12,
        'XIII': 13,
        'XIV': 14,
        'XV': 15,
        'XVI': 16
    }
    return rom_num_map[chr_num]


class ChromosomeUtil:
    def calc_moving_avg(self, arr: NDArray[(Any,)], k: int) -> str:
        """
        Calculate moving average of k data points

        Returns: 
            A 1D numpy array 
        """
        assert len(arr.shape) == 1

        # Find first MA
        ma = np.array([arr[:k].mean()])

        # Iteratively find next MAs
        for i in range(arr.size - k):
            ma = np.append(ma, ma[-1] + (arr[i + k] - arr[i]) / k)

        return ma

    def get_total_bp(self, num_seq: int):
        return (num_seq - 1) * 7 + SEQ_LEN

    def plot_horizontal_line(self, y: float) -> None:
        """
        Plot a horizontal red line denoting avg
        """
        PlotUtil().plot_horizontal_line(y, 'r', 'avg')

      
class PathUtil:
    """
    Class to get path in runtime
    
    With this, we can create correct path even when this module is called from
    modules in other directories. (e.g. child directory)
    """
    
    
    @classmethod
    def get_parent_dir(self, currentframe: FrameType) -> Path:
        '''Find parent directory path in runtime'''
        return Path(inspect.getabsfile(currentframe)).parent
    
    @classmethod
    def get_figure_dir(self) -> str:
        """
        Get figure directory in runtime.
        """
        parent_dir = self.get_parent_dir(inspect.currentframe())
        return f'{parent_dir.parent.parent}/figures'

    @classmethod
    def get_data_dir(self) -> str:
        """
        Get data directory in runtime. 
        """
        parent_dir = self.get_parent_dir(inspect.currentframe())
        return f'{parent_dir.parent.parent}/data'
    
    @classmethod
    def get_hic_data_dir(self) -> str: 
        # TODO: DRY parent_dir
        parent_dir = self.get_parent_dir(inspect.currentframe())
        return f'{parent_dir.parent.parent}/hic/data'


class IOUtil:
    # TODO: Change name - SaveUtil
    def save_figure(self, path_str: Union[str, Path]) -> Path:
        # TODO: Get figure dir in here?
        path = Path(path_str)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
       
        # Set plot size 
        plt.gcf().set_size_inches(12, 6)
        
        # Save
        plt.savefig(path, dpi=200)
        
        logging.info(f'Figure saved at: {path}')
        return path 

    def save_tsv(self, df: pd.DataFrame, path_str: Union[str, Path]) -> Path:
        """Save a dataframe in tsv format"""
        path = Path(path_str)
        self.make_parent_dirs(path)
        df.to_csv(path, sep='\t', index=False, float_format='%.3f')
        
        logging.info(f'TSV file saved at: {path}')
        return path

    def make_parent_dirs(self, path_str: Union[str, Path]) -> None:
        path = Path(path_str)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)

    def append_tsv(self, df: pd.DataFrame, path_str: Union[str, Path]) -> Path:
        """Append a dataframe to a tsv if it exists, otherwise create"""
        path = Path(path_str)
        if path.is_file():
            target_df = pd.read_csv(path, sep='\t')
            pd.concat([df, target_df], join='outer', ignore_index=True)\
                .to_csv(path, sep='\t', index=False, float_format='%3f')
            return path

        return self.save_tsv(df, path_str)


class PlotUtil:
    def plot_horizontal_line(self, y: float, color: str, text: str):
        plt.axhline(y=y, color=color, linestyle='-')
        x_lim = plt.gca().get_xlim()
        plt.text(x_lim[0] + (x_lim[1] - x_lim[0]) * 0.15,
                 y,
                 text,
                 color=color,
                 ha='center',
                 va='bottom')

    def plot_vertical_line(self, x: float, color: str, text: str):
        plt.axvline(x=x, color=color, linestyle='--')
        y_lim = plt.gca().get_ylim()
        plt.text(x,
            y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75,
            text,
            color=color,
            ha='left',
            va='center')

    def plot_stacked_bar(self, data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     colors=None, grid=False, reverse=False):
        """Plots a stacked bar chart with the data and labels provided.

        Keyword arguments:
        data            -- 2-dimensional numpy array or nested list
                        containing data for each series in rows
        series_labels   -- list of series labels (these appear in
                        the legend)
        category_labels -- list of category labels (these appear
                        on the x-axis)
        show_values     -- If True then numeric value labels will 
                        be shown on each bar
        value_format    -- Format string for numeric value labels
                        (default is "{}")
        y_label         -- Label for y-axis (str)
        colors          -- List of color labels
        grid            -- If True display grid
        reverse         -- If True reverse the order that the
                        series are displayed (left-to-right
                        or right-to-left)
        """

        ny = len(data[0])
        ind = list(range(ny))

        bar_containers = []
        cum_size = np.zeros(ny)

        data = np.array(data)

        if reverse:
            data = np.flip(data, axis=1)
            category_labels = reversed(category_labels)

        for i, row_data in enumerate(data):
            color = colors[i] if colors is not None else None
            bar_containers.append(plt.bar(ind, row_data, bottom=cum_size, 
                                label=series_labels[i], color=color))
            cum_size += row_data

        if category_labels:
            plt.xticks(ind, category_labels)

        if y_label:
            plt.ylabel(y_label)

        plt.legend()

        if grid:
            plt.grid()

        if show_values:
            for bar_container in bar_containers:
                for bar in bar_container:
                    w, h = bar.get_width(), bar.get_height()
                    plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                            value_format.format(h), ha="center", 
                            va="center")
                
    def show_grid(self) -> None:
        """Invoke this function before plotting to show grid below"""
        plt.rc('axes', axisbelow=True)
        plt.grid()