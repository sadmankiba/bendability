import pandas as pd 
import numpy as np
from nptyping import NDArray
from typing import Any

from chromosome.chromosome import Chromosome
from util.util import PathUtil

# TODO: Rename conformation to threedim

class Contact:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm 
        self._res = 400
        self._matrix = self._generate_mat()
        
    def _generate_mat(self) -> NDArray[(Any, Any)]:
        """
        Contact matrix is symmetric. Contact file is a triangular matrix file.
        Three columns: row, col, intensity. 

        For example, if a chromosome is of length 5200 and we take 400
        resolution, it's entries might be 
        0 0   8 
        0 400 4.5 
        400 400 17
        ...
        4800 5200 7 
        5200 5200 15
        """
        df = self._load_contact()
        num_rows = num_cols = int(df[['row', 'col']].max().max() / self._res) + 1
        mat = np.full((num_rows, num_cols), 0)
        
        def _fill_upper_right_half_triangle():
            for i in range(len(df)):
                mat[int(df.iloc[i].row / self._res)]\
                    [int(df.iloc[i].col / self._res)] = df.iloc[i].intensity

        def _fill_lower_left_half_triangle():
            for i in range(num_rows):
                for j in range(i):
                    mat[i][j] = mat[j][i]
        
        _fill_upper_right_half_triangle()
        _fill_lower_left_half_triangle()

        return mat

    def _load_contact(self) -> pd.DataFrame:
        df = pd.read_table(f'{PathUtil.get_data_dir()}/input_data/contact/'
            f'observed_vc_400_{self._chrm.number}.txt', 
            names=['row', 'col', 'intensity'])
        return self._remove_too_high_intensity(df)

    def _remove_too_high_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df['intensity'] < 1500]