from reader import DNASequenceReader
from constants import CHRVL, SEQ_LEN

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import math 
from pathlib import Path

class ChrV:
    "Analysis of Chromosome V in yeast"
    def __init__(self):
        self.chrv_df = DNASequenceReader().get_processed_data()[CHRVL]
         
    
    def _calc_moving_avg(self, arr: np.ndarray, k: int) -> np.ndarray:
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


    def read_chrv_lib_segment(self, start: int, end: int) -> pd.DataFrame:
        """
        Read Chr V library and cut df to contain sequences in a segment.

        Returns: 
            A pandas.DataFrame containing chr V lib of selected segment. 
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil((end - SEQ_LEN + 1) / 7)

        return self.chrv_df.loc[(self.chrv_df['Sequence #'] >= first_seq_num) 
                                & (self.chrv_df['Sequence #'] <= last_seq_num), :]
        

    def plot_moving_avg(self, start: int, end: int):
        """
        Plot simple moving average of C0
        
        Args: 
            start: Start position in the chromosome 
            end: End position in the chromosome
        """
        df_sel = self.read_chrv_lib_segment(start, end)

        # Average C0 in whole chromosome
        plt.axhline(y=self.chrv_df['C0'].mean(), color='r', linestyle='-')
        
        # Plot C0 of each sequence in Chr V library
        x = (df_sel['Sequence #'] - 1) * 7 + SEQ_LEN / 2
        y = df_sel['C0']
        plt.plot(x, y, color='blue', alpha=0.2, label=1)

        # Plot mean C0 of either side of a central value
        k = [15, 25, 50]
        colors = ['green', 'red', 'black']
        alpha = [0.7, 0.8, 0.9]
        for p in zip(k, colors, alpha):
            ma = self._calc_moving_avg(y, p[0])
            plt.plot((x + ((p[0] - 1) * 7) // 2)[:ma.size], ma, color=p[1], alpha=p[2], label=p[0])
        plt.legend()
        plt.xlabel(f'Position along chromosome V')
        plt.ylabel('Moving avg. of C0')
        
        # Save figure
        ma_fig_dir = f'figures/chrv'
        if not Path(ma_fig_dir).is_dir():
            Path(ma_fig_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{ma_fig_dir}/ma_{start}_{end}.png')
        plt.show()
    

    def plot_c0_vs_dist_from_dyad(self, dist=150):
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome V

        Args: 
            dist: +-distance from dyad to plot 
        """
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[nuc_df['Chromosome ID'] == 'chrV']['Position'].tolist()
        
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
        
        # Plot avg. line
        avg_c0 = self.chrv_df['C0'].mean()
        hline = plt.axhline(y=avg_c0, color='r', linestyle='-')
        plt.text(0, avg_c0, 'average', color='r', ha='center', va='bottom')

        x = (np.arange(mean_c0.size) - mean_c0.size / 2) * 7 + 1
        plt.plot(x, mean_c0)
        plt.xlabel('Distance from dyad(bp)')
        plt.ylabel('C0')
        plt.title(f'C0 of +-{dist} bp from nuclesome dyad')
        plt.grid() 

        fig_dir = 'figures/chrv'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.savefig(f'{fig_dir}/c0_dyad_dist_{dist}.png')
        plt.show()




        