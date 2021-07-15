from reader import DNASequenceReader
from constants import CHRVL

import matplotlib.pyplot as plt 

import math


class Analysis:
    """
    Functions to analyze DNA sequence libraries
    """
    def plot_chrv_c0(self, start: int, end: int):
        """
        Plots chromosomeV c0 values in a range 
        
        Args: 
            start: Start position in the chromosome 
            end: End position in the chromosome
        """
        first_seq_num = math.ceil(start / 7)
        last_seq_num = math.ceil(end / 7)
        
        reader = DNASequenceReader()
        all_lib = reader.get_processed_data()
        df = all_lib[CHRVL]
        df = df.loc[df['Sequence #'] >= first_seq_num & df['Sequence #'] <= last_seq_num, :]
        plt.plot(df['Sequence #'] * 7, df['C0'], linestyle='-', color='k')
        plt.xlabel('Position along Chromosome V')
        plt.ylabel('C0')
        plt.savefig(f'figures/chrv/{start}_{end}.png')
        plt.show()
