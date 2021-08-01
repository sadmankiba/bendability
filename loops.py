from scipy.ndimage.measurements import label
from reader import DNASequenceReader
from constants import CHRVL, CHRV_TOTAL_BP
from chromosome import Chromosome

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
    def __init__(self, chr: Chromosome):
        self._loop_file = f'data/input_data/loops/merged_loops_res_100_200_400_chr{chr._chr_num}.bedpe'    
        self._chr = chr


    def _read_loops(self) -> pd.DataFrame:
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: [res, start, end]
        """
        df = pd.read_table(self._loop_file, skiprows = [1])
        # TODO: Exclude same loops
        return df.assign(res = lambda df: df['x2'] - df['x1'])\
                    .assign(start = lambda df: (df['x1'] + df['x2']) / 2)\
                    .assign(end = lambda df: (df['y1'] + df['y2']) / 2)\
                        [['res', 'start', 'end']].astype(int)
    
    # ** #
    def stat_loops(self) -> None:
        """Prints statistics of loops"""
        loop_df = self._read_loops()
        # TODO
        # Loop length histogram
            # Avg. loop length per resolution 
            # Avg. loop length 
            # Quartile of loop length 
        loop_df = self._read_loops()
        max_loop_length = 100000
        loop_df = loop_df.loc[loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()
        loop_df = loop_df.assign(length = lambda df: df['end'] - df['start'])
        loop_df['length'].plot(kind='hist')
        plt.xlabel('Loop length (bp)')
        plt.title(f'Histogram of loop length. Mean = {loop_df["length"].mean()}bp. Median = {loop_df["length"].median()}bp')
        
        # TODO: Encapsulate saving figure logic in a function
        fig_dir = 'figures/chrv/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.gcf().set_size_inches(12, 6)
        plt.savefig(f'{fig_dir}/loop_highres_hist_maxlen_{max_loop_length}.png', dpi=200)

    # ** #
    def plot_c0_in_individual_loop(self):
        loop_df = self._read_loops()

        for i in range(len(loop_df)):
            row = loop_df.iloc[i]
            # TODO: -150% to +150% of loop. Vertical line = loop anchor
            self._chr.plot_moving_avg(row['start'], row['end'])
            plt.ylim(-0.7, 0.7)
            plt.xlabel(f'Position along Chromosome {self._chr._chr_num}')
            plt.ylabel('Intrinsic Cyclizability')
            plt.title(f'C0 in loop between {row["start"]}-{row["end"]}. Found with resolution: {row["res"]}.')

            # Save figure
            loop_fig_dir = f'figures/chromosome/{self._chr._chr_num}_{self._chr._c0_type}/loops/{row["res"]}'
            if not Path(loop_fig_dir).is_dir():
                Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
            
            plt.savefig(f'{loop_fig_dir}/{row["start"]}_{row["end"]}.png')


    def _plot_mean_across_loops(self, total_perc: int, chr_spread: np.ndarray, val_type: str) -> None:
        """
        Underlying plotter to plot mean across loops
        
        Plots mean C0 or mean nuc. occupancy. Does not add labels. 
        """
        loop_df = self._read_loops()

        # Filter loops by length
        max_loop_length = 100000
        loop_df = loop_df.loc[loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()
        
        def _find_value_in_loop(row: pd.Series) -> np.ndarray:
            """
            Find value from start to end considering total percentage. 
            
            Returns:
                A 1D numpy array. If value can't be calculated for whole total percentage 
                an empty array of size 0 is returned. 
            """
            start_pos = int(row['start'] + (row['end'] - row['start']) * (1 - total_perc / 100) / 2)
            end_pos = int(row['end'] + (row['end'] - row['start']) * (total_perc / 100 - 1) / 2)
            
            if start_pos < 0 or end_pos > self._chr._total_bp - 1: 
                print(f'Excluding loop: ({row["start"]}-{row["end"]})!')
                return np.empty((0,))

            return chr_spread[start_pos: end_pos]
        
        assert _find_value_in_loop(pd.Series({'start': 30, 'end': 50})).size == int(20 * total_perc / 100)
        assert _find_value_in_loop(pd.Series({'start': 50, 'end': 30})).size == 0

        # Find C0 along loop. Resize. Take mean.
        value_in_loops: pd.Series = loop_df.apply(_find_value_in_loop, axis=1)
        value_in_loops = pd.Series(list(filter(lambda arr: arr.size != 0, value_in_loops)))
        
        resize_multiple = 10
        value_in_loops = pd.Series(list(map(lambda arr: resize(arr, ((total_perc + 1) * resize_multiple, )), value_in_loops)))
        mean_val = np.array(value_in_loops.tolist()).mean(axis=0)
        
        plt.close()
        plt.clf()

        # Plot mean value
        x = np.arange((total_perc + 1) * resize_multiple) / resize_multiple - (total_perc - 100) / 2
        plt.plot(x, mean_val, color='tab:blue')
        self._chr.plot_horizontal_line(chr_spread.mean())
        plt.grid()
        
        y_lim = plt.gca().get_ylim()
        
        # Plot anchor lines
        if total_perc >= 100:
            for pos in [0, 100]:
                plt.axvline(x=pos, color='tab:green', linestyle='--')
                plt.text(pos, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, 'anchor', color='tab:green', ha='left', va='center')

        # Plot center line
        plt.axvline(x=50, color='tab:orange', linestyle='--')
        plt.text(50, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, 'center', color='tab:orange', ha='left', va='center')

        # Label plot
        plt.xlabel('Position along loop (percentage)')
        plt.ylabel(val_type)
        plt.title(f'Mean {self._chr._c0_type} {val_type} along chromosome {self._chr._chr_num} loop ({x[0]}% to {x[-1]}% of loop length)')
        
        # Save plot
        fig_dir = f'figures/chromosome/{self._chr._chr_num}_{self._chr._c0_type}/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
       
        plt.gcf().set_size_inches(12, 6)
        plt.savefig(f'{fig_dir}/mean_{val_type}_total_loop_perc_{total_perc}_maxlen_{max_loop_length}.png', dpi=200)

        
    # *** #    
    def plot_mean_c0_across_loops(self, total_perc=150) -> None:
        """
        Plot mean C0 across total loop in found loops in chr V

        Args: 
            total_perc: Total percentage of loop length to consider 
        """
        self._plot_mean_across_loops(total_perc, self._chr.spread_c0_balanced(), 'c0')
        
        
    def plot_c0_around_individual_anchor(self):
        loop_df = self._read_loops()
        
        for i in range(len(loop_df)):
            for col in ['start', 'end']:
                # Plot C0
                a = loop_df.iloc[i][col]
                self._chr.plot_moving_avg(a - lim, a + lim)
                plt.ylim(-0.7, 0.7)
                plt.xticks(ticks=[a - lim, a, a + lim], labels=[-lim, 0, +lim])
                plt.xlabel(f'Distance from loop anchor')
                plt.ylabel('Intrinsic Cyclizability')
                plt.title(f'C0 around chromosome {self._chr._chr_num} loop {col} anchor at {a}bp. Found with res {loop_df.iloc[i]["res"]}')

                # Save figure
                loop_fig_dir = f'figures/chromosome/{self._chr._chr_num}_{self._chr._c0_type}/loops/{loop_df.iloc[i]["res"]}'
                if not Path(loop_fig_dir).is_dir():
                    Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)  
                plt.savefig(f'{loop_fig_dir}/anchor_{col}_{a}.png')
        
        
    # *** #
    def plot_c0_around_anchor(self, lim=500):
        """Plot C0 around loop anchor points"""
        # TODO: Distance from loop anchor : percentage
        loop_df = self._read_loops()
        
        chrv_c0_spread = self._chr.spread_c0_balanced()
        
        def mean_around_anchors(anchors: np.ndarray) -> np.ndarray:
            """Calculate mean C0 at bp-resolution around anchors"""
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
        plt.plot(x, mean_c0_start, color='tab:green', label='start')
        plt.plot(x, mean_c0_end, color='tab:orange', label='end')
        plt.plot(x, mean_c0_all, color='tab:blue', label='all')
        self._chr.plot_avg()

        plt.legend()
        plt.grid()
        plt.xlabel('Distance from loop anchor(bp)')
        plt.ylabel('C0')
        plt.title(f'Mean {self._chr._c0_type} C0 around anchor points. Considering start, end and all anchors.')

        fig_dir = f'figures/chromosome/{self._chr._chr_num}_{self._chr._c0_type}/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)

        plt.gcf().set_size_inches(12, 6)
        plt.savefig(f'{fig_dir}/mean_c0_loop_hires_anchor_dist_{lim}_balanced.png', dpi=200)
        

    def plot_mean_nuc_occupancy_across_loops(self, total_perc=150) -> None:
        self._plot_mean_across_loops(total_perc, self._chr.get_nucleosome_occupancy(), 'nucleosome_occupancy')



        
            

