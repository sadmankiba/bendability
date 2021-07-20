from reader import DNASequenceReader
from constants import CHRVL, SEQ_LEN

import matplotlib.pyplot as plt 
import numpy as np

import math 
from pathlib import Path

class ChrV:
    "Analysis of Chromosome V in yeast"
    
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

    
    def plot_moving_avg(self, start: int, end: int):
        """
        Plot simple moving average of C0
        
        Args: 
            start: Start position in the chromosome 
            end: End position in the chromosome
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil((end - SEQ_LEN + 1) / 7)

        reader = DNASequenceReader()
        all_lib = reader.get_processed_data()
        chrv_df = all_lib[CHRVL]
        df_sel = chrv_df.loc[(chrv_df['Sequence #'] >= first_seq_num) 
                                & (chrv_df['Sequence #'] <= last_seq_num), :]

        # Average C0 in whole chromosome
        plt.axhline(y=chrv_df['C0'].mean(), color='r', linestyle='-')
        
        # Plot mean of data of either side of a central value
        x = (df_sel['Sequence #'] - 1) * 7 + SEQ_LEN / 2
        y = df_sel['C0']

        plt.plot(x, y, color='blue', alpha=0.2, label=1)
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
        