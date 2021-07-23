from scipy.ndimage.measurements import label
from reader import DNASequenceReader
from constants import CHRVL, CHRV_TOTAL_BP
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

    # ** #
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

    # *** #
    def plot_c0_vs_total_loop(self, total_perc=150):
        """
        Plot C0 in total loop

        Args: 
            total_perc: Total percentage of loop length to consider 
        """
        loop_df = self._read_loops()

        # Filter loops by length
        max_loop_length = 100000
        loop_df = loop_df.loc[loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()
        
        # Get spread C0
        chrv = ChrV()
        chrv_c0_spread = chrv.spread_c0_balanced()
        
        def _find_loop_c0(row: pd.Series) -> np.ndarray:
            """
            Find C0 from start to end considering total percentage. 
            
            Returns:
                A 1D numpy array. If C0 can't be calculated for whole total percentage 
                an empty array of size 0 is returned. 
            """
            start_pos = int(row['start'] + (row['end'] - row['start']) * (100 - total_perc) / 2)
            end_pos = int(row['end'] + (row['end'] - row['start']) * (total_perc - 100) / 2)
            
            if start_pos < 0 or end_pos > CHRV_TOTAL_BP - 1: 
                print(f'Excluding loop: ({row["start"]}-{row["end"]})!')
                return np.empty((0,))

            return chrv_c0_spread[start_pos: end_pos]
        
        assert _find_loop_c0(pd.Series({'start': 30, 'end': 50})).size == 20 * total_perc / 100
        assert _find_loop_c0(pd.Series({'start': 50, 'end': 30})).size == 0

        # loop_c0 = list(
        #     map(
        #         lambda i: chrv_c0_spread[
        #                     int(loop_df.iloc[i]['start'] 
        #                         + (loop_df.iloc[i]['end'] - loop_df.iloc[i]['start']) * (100 - total_perc) / 2)  
        #                     : int(loop_df.iloc[i]['end'] 
        #                         + (loop_df.iloc[i]['end'] - loop_df.iloc[i]['start']) * (total_perc - 100) / 2)
        #                 ],
        #         range(len(loop_df))
        #     )
        # )

        # loop_c0 = list(map(lambda arr: resize(arr, (total_perc + 1,)), loop_c0))
        
        # Find mean C0 along loop 
        loop_c0: pd.Series = loop_df.apply(_find_loop_c0, axis=1)
        loop_c0 = pd.Series(list(filter(lambda arr: arr.size != 0, loop_c0)))
        print(loop_c0.shape)
        
        mean_c0 = np.array(loop_c0.tolist()).mean(axis=0)
        
        # Plot c0
        x = np.arange(total_perc + 1) - (total_perc - 100) / 2
        plt.plot(x, mean_c0, color='tab:blue')
        chrv.plot_avg()
        plt.grid()
        
        y_lim = plt.gca().get_ylim()
        
        # Plot anchor lines
        if total_perc >= 100:
            for pos in [0, 100]:
                plt.axvline(x=pos, color='tab:green', linestyle='--')
                plt.text(pos, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, 'anchor', color='tab:green', ha='left', va='center')

        # Plot center line
        plt.axvline(x=50, color='tab:orange', linestyle='--')
        plt.text(0, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, 'center', color='tab:orange', ha='left', va='center')

        # Label plot
        plt.xlabel('Position along loop (percentage)')
        plt.ylabel('C0')
        plt.title(f'C0 along loop ({x[0]}% to {x[-1]}% of loop length)')
        
        # Save plot
        fig_dir = 'figures/chrv/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.gcf().set_size_inches(12, 6)
        plt.savefig(f'{fig_dir}/c0_total_loop_perc_{total_perc}_maxlen_{max_loop_length}.png', dpi=200)

    # *** #
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
        
        chrv_c0_spread = chrv.spread_c0_balanced()
        
        def mean_around_anchors(anchors: np.ndarray):
            return np.array(
                list(
                    map(
                        lambda a: chrv_c0_spread[a - 1 - lim: a + lim],
                        anchors
                    )
                )
            ).mean(axis=0)

        anchors = np.concatenate((loop_df['start'].to_numpy(), loop_df['end'].to_numpy()))
        mean_c0_start = mean_around_anchors(loop_df['start'].to_numpy())
        mean_c0_end = mean_around_anchors(loop_df['end'].to_numpy())
        mean_c0_all = mean_around_anchors(anchors)

        plt.close()
        plt.clf()

        x = np.arange(2 * lim + 1) - lim
        plt.plot(x, mean_c0_start, color='tab:blue', label='start')
        plt.plot(x, mean_c0_end, color='tab:green', label='end')
        plt.plot(x, mean_c0_all, color='tab:orange', label='all')
        chrv.plot_avg()

        plt.legend()
        plt.grid()
        plt.xlabel('Distance from loop anchor(bp)')
        plt.ylabel('C0')

        fig_dir = 'figures/chrv/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)

        plt.gcf().set_size_inches(12, 6)
        plt.savefig(f'{fig_dir}/c0_loop_hires_anchor_dist_{lim}_balanced.png', dpi=200)
        



        
            

