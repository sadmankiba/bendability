from __future__ import annotations

from reader import DNASequenceReader
from constants import CHRVL, SEQ_LEN, CHRV_TOTAL_BP, CHRVL_LEN
from meuseum_mod.evaluation import Evaluation
from custom_types import YeastChrNum
from util import IOUtil

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

import math 
from pathlib import Path
import time
from typing import IO, Literal, Union



class ChromosomeUtil:
    def calc_moving_avg(self, arr: np.ndarray, k: int) -> np.ndarray:
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
        plt.axhline(y=y, color='r', linestyle='-')
        x_lim = plt.gca().get_xlim()
        plt.text(x_lim[0] + (x_lim[1] - x_lim[0]) * 0.15, y, 'avg', color='r', ha='center', va='bottom')
        
SpreadType = Literal['mean7', 'mean_cover', 'weighted', 'single']

class Spread:
    """Spread C0 at each bp from C0 of 50-bp sequences at 7-bp resolution"""
    
    def __init__(self, seq_c0_res7: np.ndarray, chr_id: Union[YeastChrNum, Literal['VL']]):
        """
        Construct a spread object
        
        Args:
            seq_c0_res7: C0 of 50-bp sequences at 7-bp resolution  
            chr_id: Chromosome ID
        """
        self._seq_c0_res7 = seq_c0_res7
        self._total_bp = ChromosomeUtil().get_total_bp(seq_c0_res7.size)
        self._chr_id = chr_id


    def _covering_sequences_at(self, pos: int) -> np.ndarray:
        """
        Find covering 50-bp sequences in chr library of a bp in chromosome.
        
        Args: 
            pos: position in chr (1-indexed)
        
        Returns:
            A numpy 1D array containing sequence numbers of chr library 
            in increasing order
        """
        if pos < SEQ_LEN:   # 1, 2, ... , 50
            arr = np.arange((pos + 6) // 7) + 1
        elif pos > self._total_bp - SEQ_LEN:   # For chr V, 576822, ..., 576871
            arr = -(np.arange((self._total_bp - pos + 1 + 6) // 7)[::-1]) + self._seq_c0_res7.size 
        elif pos % 7 == 1:  # For chr V, 50, 57, 64, ..., 576821
            arr = np.arange(8) + (pos - SEQ_LEN + 7) // 7
        else: 
            arr = np.arange(7) + (pos - SEQ_LEN + 14) // 7
        
        start_pos = (arr - 1) * 7 + 1
        end_pos = start_pos + SEQ_LEN - 1
        assert np.all(pos >= start_pos) and np.all(pos <= end_pos)

        return arr


    def _mean_of_7(self) -> np.ndarray: 
        """
        Determine C0 at each bp by taking mean of 7 covering sequences
        
        If 8 sequences cover a bp, first 7 considered. If less than 7 seq
        cover a bp, nearest 7-seq mean is used. 
        """
        saved_data = Path(f'data/generated_data/spread/spread_c0_mean7_{self._chr_id}.tsv')
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep='\t')['c0_mean7'].to_numpy()

        mvavg = ChromosomeUtil().calc_moving_avg(self._seq_c0_res7, 7)
        spread_mvavg = np.vstack((mvavg, mvavg, mvavg, mvavg, mvavg, mvavg, mvavg)).ravel(order='F')
        full_spread = np.concatenate((
            np.full((42,), spread_mvavg[0]), 
            spread_mvavg,
            np.full((43,), spread_mvavg[-1])
        ))
        assert full_spread.shape == (self._total_bp,)
        
        IOUtil().save_tsv(
            pd.DataFrame({'position': np.arange(self._total_bp) + 1, 'c0_mean7': full_spread}),
            saved_data
        )
        return full_spread 
         

    def _mean_of_covering_seq(self) -> np.ndarray:
        """Determine C0 at each bp by average of covering 50-bp sequences around"""
        # TODO: HOF to wrap check and save data?
        saved_data = Path(f'data/generated_data/spread/spread_c0_balanced_{self._chr_id}.tsv')
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep='\t')['c0_balanced'].to_numpy()
        
        def balanced_c0_at(pos) -> float:
            seq_indices = self._covering_sequences_at(pos) - 1
            return self._df['C0'][seq_indices].mean()
        
        t = time.time()
        res = np.array(list(map(balanced_c0_at, np.arange(self._total_bp) + 1)))
        print('Calculation of spread c0 balanced:', time.time() - t, 'seconds.')
        
        # Save data
        if not saved_data.parents[0].is_dir():
            saved_data.parents[0].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'position': np.arange(self._total_bp) + 1, 'c0_balanced': res})\
            .to_csv(saved_data, sep='\t', index=False)

        return res


    def _weighted_covering_seq(self) -> np.ndarray:
        """Determine C0 at each bp by weighted average of covering 50-bp sequences around"""
        saved_data = Path(f'data/generated_data/spread/spread_c0_weighted_{self._chr_id}.tsv')
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep='\t')['c0_weighted'].to_numpy()

        def weights_for(size: int) -> list[int]:
            # TODO: Use short algorithm
            if size == 1: 
                return [1]
            elif size == 2: 
                return [1, 1]
            elif size == 3: 
                return [1, 2, 1]
            elif size == 4: 
                return [1, 2, 2, 1]
            elif size == 5: 
                return [1, 2, 3, 2, 1]
            elif size == 6: 
                return [1, 2, 3, 3, 2, 1]
            elif size == 7: 
                return [1, 2, 3, 4, 3, 2, 1]
            elif size == 8: 
                return [1, 2, 3, 4, 4, 3, 2, 1]
            
        def weighted_c0_at(pos) -> float:
            seq_indices = self._covering_sequences_at(pos) - 1
            c0s = self.chrv_df['C0'][seq_indices].to_numpy()
            return np.sum(c0s * weights_for(c0s.size)) / sum(weights_for(c0s.size))

        t = time.time()
        res = np.array(list(map(weighted_c0_at, np.arange(self._total_bp) + 1)))
        print(print('Calculation of spread c0 weighted:', time.time() - t, 'seconds.'))
        
        # Save data
        if not saved_data.parents[0].is_dir():
            saved_data.parents[0].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'position': np.arange(self._total_bp) + 1, 'c0_weighted': res})\
            .to_csv(saved_data, sep='\t', index=False)

        return res 


    def _from_single_seq(self) -> np.ndarray:
        """Determine C0 at each bp by spreading C0 of a 50-bp seq to position 22-28"""
        c0_arr = self._seq_c0_res7
        spread = np.concatenate((
            np.full((21,), c0_arr[0]), 
            np.vstack(([c0_arr] * 7)).ravel(order='F'),
            np.full((22,), c0_arr[-1])
        ))
        # assert spread.size == self._total_bp
        
        return spread


    def get_spread(self, spread_str: SpreadType) -> np.ndarray:
        if spread_str == 'mean7':
            return self._mean_of_7()
        elif spread_str == 'mean_cover':
            return self._mean_of_covering_seq()
        elif spread_str == 'weighted':
            return self._weighted_covering_seq()
        elif spread_str == 'single':
            return self._from_single_seq()

YeastChrNumList = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 
                        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']

class Chromosome:
    "Analysis of Chromosome in yeast"

    def __init__(self, chr_id: Union[YeastChrNum, Literal['VL']], 
                    spread_str : SpreadType = 'mean7'):
        """
        Create a Chromosome object 

        Args: 
            chr_id: For Roman Numbers, predicted C0 is used. 'VL' represents 
                chr V library of bendability data.
            spread_str: Which type of spread to use. 
        """
        # TODO: _chr_num, _c0_type should be public (used in Loops)
        self._chr_num = 'V' if chr_id == 'VL' else chr_id
        self._chr_id = chr_id
        self._c0_type = 'actual' if chr_id == 'VL' else 'predicted'
        self._df = (DNASequenceReader().get_processed_data()[CHRVL] 
            if chr_id == 'VL' else self._get_chr_prediction())
        # TODO: Don't keep _chr_util attribute
        self._chr_util = ChromosomeUtil()  
        self._total_bp = self._chr_util.get_total_bp(len(self._df))
        self.spread_str = spread_str
         

    def _get_chr_prediction(self):
        """Read predicted C0 of a yeast chromosome by meuseum model"""

        saved_predict_data = Path(f'data/generated_data/predictions/chr{self._chr_num}_pred.tsv')
        if saved_predict_data.is_file():
            return pd.read_csv(saved_predict_data, sep='\t')
        
        df = DNASequenceReader().read_yeast_genome(self._chr_num)
        predict_df = Evaluation().predict(df).rename(columns = {'c0_predict': 'C0'})
        
        # Save data
        IOUtil().save_tsv(predict_df, saved_predict_data)
        # if not saved_predict_data.parents[0].is_dir():
        #     saved_predict_data.parents[0].mkdir(parents=True, exist_ok=True)
        # predict_df.to_csv(saved_predict_data, sep='\t', index=False)
        
        return predict_df


    def read_chr_lib_segment(self, start: int, end: int) -> pd.DataFrame:
        """
        Get sequences in library of a chromosome segment.

        Returns: 
            A pandas.DataFrame containing chr library of selected segment. 
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil((end - SEQ_LEN + 1) / 7)

        return self._df.loc[(self._df['Sequence #'] >= first_seq_num) 
                                & (self._df['Sequence #'] <= last_seq_num), :]
    

    def plot_avg(self) -> None: 
        """
        Plot a horizontal red line denoting avg. C0 in whole chromosome
        
        Best to draw after x limit is set.
        """
        self._chr_util.plot_horizontal_line(self._df['C0'].mean())
        

    def plot_moving_avg(self, start: int, end: int) -> None:
        """
        Plot C0, a few moving averages of C0 and nuc. centers in a segment of chr.

        Does not give labels so that custom labels can be given after calling this
        function.  
        
        Args: 
            start: Start position in the chromosome 
            end: End position in the chromosome
        """
        df_sel = self.read_chr_lib_segment(start, end)

        plt.close()
        plt.clf()
        
        # Plot C0 of each sequence in Chr library
        x = (df_sel['Sequence #'] - 1) * 7 + SEQ_LEN / 2
        y = df_sel['C0'].to_numpy()
        plt.plot(x, y, color='blue', alpha=0.2, label=1)

        # Plot mean C0 of either side of a central value
        k = [10] #, 25, 50]
        colors = ['green'] #, 'red', 'black']
        alpha = [0.7] #, 0.8, 0.9]
        for p in zip(k, colors, alpha):
            ma = self._chr_util.calc_moving_avg(y, p[0])
            plt.plot((x + ((p[0] - 1) * 7) // 2)[:ma.size], ma, color=p[1], alpha=p[2], label=p[0])
        
        self.plot_avg()

        # Find and plot nuc. centers
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[
                        nuc_df['Chromosome ID'] == f'chr{self._chr_num}'
                    ]['Position'].to_numpy()
        centers = centers[centers > start]
        centers = centers[centers < end]
        
        for c in centers:
            plt.axvline(x=c, color='grey', linestyle='--')

        plt.legend()
        plt.grid()

    # *** #
    def plot_c0(self, start: int, end: int) -> None:
        """Plot C0, moving avg., nuc. centers of a segment in chromosome
        and add appropriate labels.   
        """
        self.plot_moving_avg(start, end)

        plt.ylim(-0.8, 0.6)
        plt.xlabel(f'Position along chromosome {self._chr_num}')
        plt.ylabel('Moving avg. of C0')
        plt.title(f"C0, 10-bp moving avg. of C0 and nuclesome centers in Chr {self._chr_num} ({start}-{end})")
        
        # Save figure
        plt.gcf().set_size_inches(12, 6)
        ma_fig_dir = f'figures/chromosome/{self._chr_id}'
        if not Path(ma_fig_dir).is_dir():
            Path(ma_fig_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{ma_fig_dir}/ma_{start}_{end}.png', dpi=200)
        plt.show()


    def get_spread(self) -> np.ndarray:
        return Spread(self._df['C0'].values, self._chr_id).get_spread(self.spread_str)
    

    
        