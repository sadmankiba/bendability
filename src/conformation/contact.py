from typing import Any
from pathlib import Path 

import pandas as pd 
import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt

from chromosome.chromosome import Chromosome
from util.util import PathUtil, IOUtil

# TODO: Rename conformation to threedim

class Contact:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm 
        self._res = 400
        # TODO: Save matrix
        self._matrix = self._generate_mat()
    
    def show(self) -> Path:
        # TODO: Image from single color. Not RGB.
        img = np.zeros((self._matrix.shape[0], self._matrix.shape[1], 3))
        MIN_INTENSITY_FOR_MAX_COLOR = 100
        MAX_PIXEL_INTENSITY = 255
        img[:,:,0] = (self._matrix / MIN_INTENSITY_FOR_MAX_COLOR * MAX_PIXEL_INTENSITY).astype(int)
        img[img > MAX_PIXEL_INTENSITY] = MAX_PIXEL_INTENSITY
       
        plt.imshow(img, interpolation='nearest')
        return IOUtil().save_figure(
            f'{PathUtil.get_figure_dir()}/contact/observed_vc_{self._res}_{self._chrm.number}.png')
        
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
        saved_contact = Path(
            f"{PathUtil.get_data_dir()}/generated_data/contact"
            f"/observed_vc_{self._res}_{self._chrm.number}.npy"
        )

        if saved_contact.is_file():
            return np.load(saved_contact)

        df = self._load_contact()
        df[['row', 'col']] = df[['row', 'col']] / self._res
        num_rows = num_cols = int(df[['row', 'col']].max().max()) + 1
        mat: NDArray = np.full((num_rows, num_cols), 0)
        
        def _fill_upper_right_half_triangle():
            for i in range(len(df)):
                elem = df.iloc[i]
                mat[int(elem.row)][int(elem.col)] = elem.intensity

        def _fill_lower_left_half_triangle(mat) -> NDArray[(Any, Any)]:
            ll = np.copy(mat.transpose())
            for i in range(num_rows):
                ll[i][i] = 0
            
            mat += ll
            return mat

        _fill_upper_right_half_triangle()
        mat = _fill_lower_left_half_triangle(mat)
        
        np.save(saved_contact, mat)
        return mat

    def _load_contact(self) -> pd.DataFrame:
        df = pd.read_table(f'{PathUtil.get_data_dir()}/input_data/contact/'
            f'observed_vc_400_{self._chrm.number}.txt', 
            names=['row', 'col', 'intensity'])
        return self._remove_too_high_intensity(df)

    MAX_INTENSITY = 1500

    def _remove_too_high_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df['intensity'] < 1500]