from reader import DNASequenceReader
from constants import CHRVL

import matplotlib.pyplot as plt 
import pandas as pd

import math
from pathlib import Path 


class Loops:
    """
    Functions to analyze DNA sequence libraries
    """
    def __init__(self, loop_file: str):
        self._loop_file = loop_file    


    def _read_loops(self):
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: resolution, start, end
        """
        df = pd.read_table(self._loop_file, skiprows = [1])
        return df.assign(res = lambda x: x['x2'] - x['x1'])\
                [['res', 'x2', 'y1']]\
                    .rename(columns={'x2': 'start', 'y1': 'end'})
        


    def plot_chrv_c0(self, start: int, end: int, res: int):
        """
        Plots chromosomeV c0 values in a loop 
        
        Args: 
            start: Start position of loop in the chromosome 
            end: End position of loop in the chromosome
            res: Resolution used to find this loop
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil(end / 7)
        
        reader = DNASequenceReader()
        all_lib = reader.get_processed_data()
        df = all_lib[CHRVL]
        df = df.loc[(df['Sequence #'] >= first_seq_num) & (df['Sequence #'] <= last_seq_num), :]
        plt.close()
        plt.clf()
        plt.plot(df['Sequence #'] * 7, df['C0'], linestyle='-', color='k')
        plt.xlabel(f'Position along Chromosome V')
        plt.ylabel('Intrinsic Cyclizability')
        plt.title(f'C0 in loop between {start}-{end}. Found with resolution: {res}.')
        
        # Save figure
        loop_fig_dir = f'figures/chrv_loops/{res}'
        if not Path(loop_fig_dir).is_dir():
            Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{loop_fig_dir}/{start}_{end}.png')


    def plot_chrv_c0_in_loops(self):
        loop_df = self._read_loops()

        for i in range(len(loop_df)):
            row = loop_df.iloc[i]
            self.plot_chrv_c0(row['start'], row['end'], row['res']) 
