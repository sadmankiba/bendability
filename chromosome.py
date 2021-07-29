from __future__ import annotations

from reader import DNASequenceReader
from constants import CHRVL, SEQ_LEN, CHRV_TOTAL_BP, CHRVL_LEN
from meuseum_mod.evaluation import Evaluation

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

import math 
from pathlib import Path
import time
from typing import Literal, Union

YeastChrNum = Literal['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 
                        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']


class ChromosomeUtil:
    def calc_moving_avg(self, arr: np.ndarray, k: int) -> np.ndarray:
        """
        Calculate moving average of k data points 
        """
        assert len(arr.shape) == 1

        # Find first MA
        ma = np.array([arr[:k].mean()])
        
        # Iteratively find next MAs
        for i in range(arr.size - k):
            ma = np.append(ma, ma[-1] + (arr[i + k] - arr[i]) / k)
        
        return ma


class Chromosome:
    "Analysis of Chromosome in yeast"
    # TODO: Subclass Chr based on actual vs. predicted ?

    def __init__(self, chr_num: Union[YeastChrNum, Literal['VL']]):
        # TODO: chr_num and c0_type should be public (used in Loops)
        self._chr_num = 'V' if chr_num == 'VL' else chr_num
        self._c0_type = 'actual' if chr_num == 'VL' else 'predicted'
        self._df = (DNASequenceReader().get_processed_data()[CHRVL] 
            if chr_num == 'VL' else self._get_chr_prediction(chr_num))
        self._total_bp = (len(self._df) - 1) * 7 + SEQ_LEN
        self._chr_util = ChromosomeUtil()
         

    def _get_chr_prediction(self, chr_num: YeastChrNum):
        """Read predicted C0 of a yeast chromosome by meuseum model"""

        saved_predict_data = Path(f'data/generated_data/predictions/chr{chr_num}_pred.tsv')
        if saved_predict_data.is_file():
            return pd.read_csv(saved_predict_data, sep='\t')
        
        df = DNASequenceReader().read_yeast_genome(7)
        predict_df = Evaluation().predict(df).rename(columns = {'c0_predict': 'C0'})
        
        # Save data
        if not saved_predict_data.parents[0].is_dir():
            saved_predict_data.parents[0].mkdir(parents=True, exist_ok=True)
        predict_df.to_csv(saved_predict_data, sep='\t', index=False)
        
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
            arr = -(np.arange((self._total_bp - pos + 1 + 6) // 7)[::-1]) + len(self._df) 
        elif pos % 7 == 1:  # For chr V, 50, 57, 64, ..., 576821
            arr = np.arange(8) + (pos - SEQ_LEN + 7) // 7
        else: 
            arr = np.arange(7) + (pos - SEQ_LEN + 14) // 7
        
        start_pos = (arr - 1) * 7 + 1
        end_pos = start_pos + SEQ_LEN - 1
        assert np.all(pos >= start_pos) and np.all(pos <= end_pos)

        return arr

    
    def plot_avg(self) -> None: 
        """
        Plot a horizontal red line denoting avg. C0 in whole chromosome
        
        Best to draw after x limit is set.
        """
        plt.axhline(y=self._df['C0'].mean(), color='r', linestyle='-')
        x_lim = plt.gca().get_xlim()
        plt.text(x_lim[0] + (x_lim[1] - x_lim[0]) * 0.15, self._df['C0'].mean(), 'avg', color='r', ha='center', va='bottom')
        

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
        ma_fig_dir = f'figures/chromosome/{self._chr_num}_{self._c0_type}'
        if not Path(ma_fig_dir).is_dir():
            Path(ma_fig_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{ma_fig_dir}/ma_{start}_{end}.png', dpi=200)
        plt.show()
    

    def _plot_c0_vs_dist_from_dyad(self, x: np.ndarray, y: np.ndarray, dist: int, spread_str: str) -> None:
        """Underlying plotter of c0 vs dist from dyad"""
        # Plot C0
        plt.plot(x, y, color='tab:blue')
        
        # Highlight nuc. end positions and dyad
        y_lim = plt.gca().get_ylim()
        for p in [-73, 73]:
            plt.axvline(x=p, color='tab:green', linestyle='--')
            plt.text(p, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, f'{p}bp', color='tab:green', ha='left', va='center')

        plt.axvline(x=0, color='tab:orange', linestyle='--')
        plt.text(0, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, f'dyad', color='tab:orange', ha='left', va='center')

        self.plot_avg()
        plt.grid()

        plt.xlabel('Distance from dyad(bp)')
        plt.ylabel('C0')
        plt.title(f'C0 of +-{dist} bp from nuclesome dyad')
        
        # Save figure
        fig_dir = f'figures/chromosome/{self._chr_num}_{self._c0_type}'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.savefig(f'{fig_dir}/c0_dyad_dist_{dist}_{spread_str}.png')


    def plot_c0_vs_dist_from_dyad_no_spread(self, dist=150) -> None:
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome V from 50-bp sequence C0

        Currently, giving horizontally shifted graph. (incorrect)
         
        Args: 
            dist: +-distance from dyad to plot (1-indexed)
        """
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[nuc_df['Chromosome ID'] == f'chr{self._chr_num}']['Position'].tolist()
        
        # Read C0 of -dist to +dist sequences
        c0_at_nuc: list[list[float]] = list(
            map(
                lambda c: self.read_chrv_lib_segment(c - dist, c + dist)['C0'].tolist(), 
                centers
            )
        )

        # Make lists of same length
        min_len = min(map(len, c0_at_nuc))
        max_len = max(map(len, c0_at_nuc))
        assert max_len - min_len <= 1
        c0_at_nuc = list(
            map(
                lambda l: l[:min_len],
                c0_at_nuc
            )
        ) 
        mean_c0 = np.array(c0_at_nuc).mean(axis=0)
        x = (np.arange(mean_c0.size) - mean_c0.size / 2) * 7 + 1

        self._plot_c0_vs_dist_from_dyad(x, mean_c0, dist, 'no_spread')
        

    def spread_c0(self) -> np.ndarray:
        """Determine C0 at each bp by spreading C0 of a 50-bp seq to position 22-28"""
        c0_arr = self._df['C0'].to_numpy()
        spread = np.concatenate((
            np.full((21,), c0_arr[0]), 
            np.vstack(([c0_arr] * 7)).ravel(order='F'),
            np.full((22,), c0_arr[-1])
        ))
        assert spread.size == self._total_bp
        
        return spread

    
    def spread_c0_balanced(self) -> np.ndarray:
        """Determine C0 at each bp by average of covering 50-bp sequences around"""
        # TODO: spread C0 -> separate class

        saved_data = Path(f'data/generated_data/chromosome/{self._chr_num}_{self._c0_type}/spread_c0_balanced.tsv')
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


    def spread_c0_weighted(self) -> np.ndarray:
        """Determine C0 at each bp by weighted average of covering 50-bp sequences around"""
        saved_data = Path(f'data/generated_data/chromosome/{self._chr_num}_{self._c0_type}/spread_c0_weighted.tsv')
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

    # *** #
    def plot_c0_vs_dist_from_dyad_spread(self, dist=150) -> None:
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome by
        spreading 50-bp sequence C0

        Args: 
            dist: +-distance from dyad to plot (1-indexed) 
        """
        spread_c0 = self.spread_c0_balanced()
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[nuc_df['Chromosome ID'] == f'chr{self._chr_num}']['Position'].tolist()
        
        # Remove center positions at each end that aren't in at least dist depth
        centers = list(filter(lambda i: i > dist and i < self._total_bp - dist, centers))

        # Read C0 of -dist to +dist sequences
        c0_at_nuc: list[np.ndarray] = list(
            map(
                lambda c: spread_c0[c - 1 - dist:c + dist], 
                centers
            )
        )
        assert c0_at_nuc[0].size == 2 * dist + 1
        assert c0_at_nuc[-1].size == 2 * dist + 1
        
        x = np.arange(dist * 2 + 1) - dist
        mean_c0 = np.array(c0_at_nuc).mean(axis=0)

        self._plot_c0_vs_dist_from_dyad(x, mean_c0, dist, 'balanced')




        