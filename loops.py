from reader import DNASequenceReader
from constants import CHRVL
from chrv import ChrV

import matplotlib.pyplot as plt 
import pandas as pd
from skimage.transform import resize
import numpy as np

import math
from pathlib import Path 


class Loops:
    """
    Functions to analyze DNA sequence libraries
    """
    def __init__(self, loop_file='juicer/data/generated_data/loops/a364_loops_hires/merged_loops.bedpe'):
        """
        Args:
            loop_file: Path of .bedpe file containing loop positions generated by juicer tools
        """
        self._loop_file = loop_file    


    def _read_loops(self) -> pd.DataFrame:
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: resolution, start, end
        """
        df = pd.read_table(self._loop_file, skiprows = [1])
        # TODO: Exclude same loops
        return df.assign(res = lambda df: df['x2'] - df['x1'])\
                    .assign(start = lambda df: (df['x1'] + df['x2']) / 2)\
                    .assign(end = lambda df: (df['y1'] + df['y2']) / 2)\
                        [['res', 'start', 'end']].astype(int)
    

    def stat_loops(self) -> None:
        """Prints statistics of loops"""
        loop_df = self._read_loops()
        # TODO
        # Loop length histogram
            # Avg. loop length per resolution 
            # Avg. loop length 
            # Quartile of loop length 


    def plot_chrv_c0_in_loops(self):
        loop_df = self._read_loops()

        chrv = ChrV()
        for i in range(len(loop_df)):
            row = loop_df.iloc[i]
            # TODO: -150% to +150% of loop. Vertical line = loop anchor
            chrv.plot_moving_avg(row['start'], row['end'])
            plt.ylim(-0.7, 0.7)
            plt.xlabel(f'Position along Chromosome V')
            plt.ylabel('Intrinsic Cyclizability')
            plt.title(f'C0 in loop between {row["start"]}-{row["end"]}. Found with resolution: {row["res"]}.')

            # Save figure
            loop_fig_dir = f'figures/chrv/loops/{row["res"]}'
            if not Path(loop_fig_dir).is_dir():
                Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
            
            plt.savefig(f'{loop_fig_dir}/{row["start"]}_{row["end"]}.png')


    def plot_c0_vs_dist_from_loop_center(self):
        """
        Plot C0 vs distance from loop centers
        """
        loop_df = self._read_loops()

        # Filter loops 
        max_loop_length = 100000
        loop_df = loop_df.loc[loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()
        
        # List C0 of two times loop length around center
        loop_df = loop_df.assign(center = (loop_df['start'] + loop_df['end']) / 2)
        chrv = ChrV()
        # loop_df = loop_df.assign(
        #     C0 = lambda row: chrv.read_chrv_lib_segment(
        #         row['start'] * 2 - row['center'], 
        #         row['end'] * 2 - row['center']
        #     )['C0'].to_numpy()
        # )
        loop_c0 = list(
            map(
                lambda i: chrv.read_chrv_lib_segment(
                    loop_df.loc[i]['start'] * 2 - loop_df.loc[i]['center'], 
                    loop_df.loc[i]['end'] * 2 - loop_df.loc[i]['center']
                )['C0'].to_numpy(),
                range(len(loop_df))
            )
        ) 

        # Convert 1D arrays to have length 201 (-100% - 100%)
        loop_c0 = list(map(lambda arr: resize(arr, (201,)), loop_c0))

        mean_c0 = np.array(loop_c0).mean(axis=0)

        # Plot avg. line
        avg_c0 = chrv.chrv_df['C0'].mean()
        hline = plt.axhline(y=avg_c0, color='r', linestyle='-')
        plt.text(0, avg_c0, 'average', color='r', ha='center', va='bottom')

        x = (np.arange(mean_c0.size) - mean_c0.size // 2)
        plt.plot(x, mean_c0)
        plt.xlabel('distance from loop center(percentage)')
        plt.ylabel('C0')

        fig_dir = 'figures/chrv'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.savefig(f'{fig_dir}/c0_loop_hires_center_dist.png')
        plt.show()


    def plot_c0_around_anchor(self, lim=2000):
        """Plot C0 around loop anchor points"""
        loop_df = self._read_loops()
        
        chrv = ChrV()
        for i in range(len(loop_df)):
            for col in ['start', 'end']:
                a = loop_df.iloc[i][col]
                chrv.plot_moving_avg(a - lim, a + lim)
                plt.ylim(-0.7, 0.7)
                plt.xticks(ticks=[a - lim, a, a + lim], labels=[-lim, 0, +lim])
                plt.xlabel(f'Distance from loop anchor')
                plt.ylabel('Intrinsic Cyclizability')
                plt.title(f'C0 around loop anchor at {a}bp. Found with res {loop_df.iloc[i]["res"]}')

                # Save figure
                loop_fig_dir = f'figures/chrv/loops/{loop_df.iloc[i]["res"]}'
                if not Path(loop_fig_dir).is_dir():
                    Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
                
                plt.savefig(f'{loop_fig_dir}/anchor_{a}.png')
        
        chrv_c0_spread = chrv._spread_c0()
        anchors = np.concatenate((loop_df['start'].to_numpy(), loop_df['end'].to_numpy()))
        mean_c0 = np.array(
            list(
                map(
                    lambda a: chrv_c0_spread[a - 1 - lim: a + lim],
                    anchors
                )
            )
        ).mean(axis=0)

        plt.close()
        plt.clf()

        x = np.arange(mean_c0.size) - mean_c0.size // 2
        plt.plot(x, mean_c0)
        plt.xlabel('Distance from loop anchor(bp)')
        plt.ylabel('C0')

        fig_dir = 'figures/chrv/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.savefig(f'{fig_dir}/c0_loop_hires_anchor_dist_{lim}.png')
        



        
            

