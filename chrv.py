from reader import DNASequenceReader
from constants import CHRVL

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
        last_seq_num = math.ceil(end / 7)

        reader = DNASequenceReader()
        all_lib = reader.get_processed_data()
        df = all_lib[CHRVL]
        df = df.loc[(df['Sequence #'] >= first_seq_num) & (df['Sequence #'] <= last_seq_num), :]
        x = df['Sequence #'] * 7
        y = df['C0']

        def calc_ma():
            pass
        plt.close()
        plt.clf()
        plt.plot(x, y, linestyle='-', color='k')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(x, y, color='black')
        plt.plot(x, y, color='blue', alpha=0.7)
        plt.plot(x, y, color='blue', alpha=0.2)
        plt.xlabel(f'Position along Chromosome V')
        plt.ylabel('Intrinsic Cyclizability')
        plt.title(f'C0 in loop between {start}-{end}.')
        
        # Save figure
        loop_fig_dir = f'figures/chrv_loops'
        if not Path(loop_fig_dir).is_dir():
            Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{loop_fig_dir}/{start}_{end}.png')
        