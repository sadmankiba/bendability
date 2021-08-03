from __future__ import annotations
from typing import IO

from custom_types import ChrId
from prediction import Prediction
from reader import DNASequenceReader
from constants import CHRVL, CHRV_TOTAL_BP
from chromosome import Chromosome, ChrIdList
from util import IOUtil

import matplotlib.pyplot as plt 
import pandas as pd
from skimage.transform import resize
import numpy as np

import math
from pathlib import Path 
import itertools
from typing import Literal


class Loops:
    """
    Abstraction of collection of loops in a single chromosome
    """
    def __init__(self, chr: Chromosome):
        self._loop_file = f'data/input_data/loops/merged_loops_res_500_chr{chr._chr_num}.bedpe'    
        self._chr = chr
        self._loop_df = self._read_loops()


    def _read_loops(self) -> pd.DataFrame:
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: [res, start, end]
        """
        df = pd.read_table(self._loop_file, skiprows = [1])
        # TODO: Exclude same loops for multiple resolutions
        def _use_middle_coordinates() -> pd.DataFrame:
            return df.assign(res = lambda df: df['x2'] - df['x1'])\
                    .assign(start = lambda df: (df['x1'] + df['x2']) / 2)\
                    .assign(end = lambda df: (df['y1'] + df['y2']) / 2)\
                        [['res', 'start', 'end']].astype(int)
        
        def _use_centroids() -> pd.DataFrame:
            return df.assign(res = lambda df: df['x2'] - df['x1'])\
                    .rename(columns={'centroid1': 'start', 'centroid2': 'end'})\
                        [['res', 'start', 'end']].astype(int)
        
        return _use_centroids().assign(len = lambda df: df['end'] - df['start'])
 

    def _get_quartile_dfs(self, df: pd.DataFrame):
        """Split a dataframe into 4 representing quartile on column len"""
        quart1, quart2, quart3 = df['len'].quantile([0.25, 0.5, 0.75]).tolist()
        return ( df.loc[df['len'] <= quart1],
            df.loc[(quart1 < df['len']) & (df['len'] <= quart2)],
            df.loc[(quart2 < df['len']) & (df['len'] <= quart3)],
            df.loc[quart3 < df['len']]
        )

    # ** #
    def stat_loops(self) -> None:
        """Prints statistics of loops"""
        loop_df = self._loop_df
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
        loop_df = self._loop_df

        for i in range(len(loop_df)):
            row = loop_df.iloc[i]
            # TODO: -150% to +150% of loop. Vertical line = loop anchor
            self._chr.plot_moving_avg(row['start'], row['end'])
            plt.ylim(-0.7, 0.7)
            plt.xlabel(f'Position along Chromosome {self._chr._chr_num}')
            plt.ylabel('Intrinsic Cyclizability')
            plt.title(f'C0 in loop between {row["start"]}-{row["end"]}. Found with resolution: {row["res"]}.')

            IOUtil().save_figure(f'figures/loop/{self._chr._chr_id}/{row["start"]}_{row["end"]}.png')
            
            # loop_fig_dir = f'figures/loops/{self._chr._chr_id}'
            # if not Path(loop_fig_dir).is_dir():
            #     Path(loop_fig_dir).mkdir(parents=True, exist_ok=True)
            
            # plt.savefig(f'{loop_fig_dir}/{row["start"]}_{row["end"]}.png')


    def _plot_mean_across_loops(self, total_perc: int, chr_spread: np.ndarray, val_type: str) -> None:
        """
        Underlying plotter to plot mean across loops
        
        Plots mean C0 or mean nuc. occupancy. Does not add labels. 
        """
        loop_df = self._loop_df

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
        
        IOUtil().save_figure(f'figures/loop/mean_{val_type}_p_{total_perc}_mxl_{max_loop_length}_{self._chr}.png')

        
    # *** #    
    def plot_mean_c0_across_loops(self, total_perc=150) -> None:
        """
        Plot mean C0 across total loop in found loops in chr V

        Args: 
            total_perc: Total percentage of loop length to consider 
        """
        self._plot_mean_across_loops(total_perc, self._chr.get_spread(), 'c0')
        
    # ** #    
    def plot_c0_around_individual_anchor(self, lim=500):
        loop_df = self._loop_df
        
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
                IOUtil().save_figure(f'figures/loop_anchor/{self._chr}/{col}_{a}.png')
        
        
    # *** #
    def plot_c0_around_anchor(self, lim=500):
        """Plot C0 around loop anchor points"""
        # TODO: Distance from loop anchor : percentage
        loop_df = self._loop_df
        
        chrv_c0_spread = self._chr.get_spread()
        
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

        IOUtil().save_figure(f'figures/loop_anchor/dist_{lim}_{self._chr}.png')
        

    def plot_mean_nuc_occupancy_across_loops(self, total_perc=150) -> None:
        self._plot_mean_across_loops(total_perc, self._chr.get_nucleosome_occupancy(), 'nuc_occ')


    def find_avg_c0(self, loop_df: pd.DataFrame=None) -> float:
        """Find average c0 of collection of loops. 

        First, average c0 of individual loops are calculated. Then, mean is
        taken over all loop averages. 

        Args: 
            loop_df: A dataframe of loops. If None, then all loops of this 
                object are considered. 
        """
        if loop_df is None:
            loop_df = self._loop_df
        chrv_c0_spread = self._chr.get_spread()
        return round(sum(
            map(
                lambda idx: chrv_c0_spread[loop_df.iloc[idx]['start'] - 1 : loop_df.iloc[idx]['end']].mean(), 
                range(len(loop_df))
            )
        ) / len(loop_df), 3)


    def find_avg_c0_in_quartile_by_len(self) -> list[float]:
        """Find average c0 of collection of loops by dividing them into
        quartiles by length"""
        quart_loop_df = self._get_quartile_dfs(self._loop_df)
        return list(map(self.find_avg_c0, quart_loop_df))


    def find_avg_c0_in_quartile_by_pos(self, loop_df : pd.DataFrame = None) -> np.ndarray:
        """Find average c0 of different positions in collection of loops
        
        Returns: 
            A 1D numpy array of size 4
        """
        if loop_df is None:
            loop_df = self._loop_df
        
        chrv_c0_spread = self._chr.get_spread()
        
        def _avg_c0_in_quartile_by_pos(row: pd.Series) -> list[float]:
            quart_pos = row.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).astype(int)
            quart_range = list(map(
                    lambda idx: (quart_pos.iloc[idx], quart_pos.iloc[idx + 1]), 
                    range(len(quart_pos) - 1)
            ))
            return list(map(lambda r: chrv_c0_spread[r[0] - 1: r[1] - 1].mean(), quart_range))
        
        result = np.round(np.mean(
            loop_df[['start', 'end']].apply(_avg_c0_in_quartile_by_pos, axis=1).tolist(), 
            axis=0
        ), 3)
        assert result.shape == (4,)
        return result


    def find_avg_c0_in_quartile_by_pos_in_quart_len(self) -> np.ndarray: 
        """Find average c0 of different positions in quartile of loops by length
        
        Returns: 
            A 1D numpy array of size 16. 
        """
        loop_df = self._loop_df
        quart1, quart2, quart3 = loop_df['len'].quantile([0.25, 0.5, 0.75]).tolist()
        quart_loop_df = self._get_quartile_dfs(loop_df)
        return np.array(list(map(self.find_avg_c0_in_quartile_by_pos, quart_loop_df))).flatten()
    

    def find_avg_around_anc(self, pos: Literal['start', 'end', 'center'], lim : int = 500, loop_df=None) -> float:
        if loop_df is None:
            loop_df = self._loop_df

        chrv_c0_spread = self._chr.get_spread()    
        
        loop_df = loop_df.assign(center = lambda df : (df['start'] + df['end']) / 2)
        
        return round(sum(
            map(
                lambda idx: chrv_c0_spread[
                    loop_df.iloc[idx][pos] - lim - 1 
                        : loop_df.iloc[idx][pos] + lim
                    ].mean(), 
                range(len(loop_df))
            )
        ) / len(loop_df), 3) 


    def find_avg_around_anc_in_quartile_by_len(self, pos: Literal['start', 'end', 'center'], lim :int = 500) -> list[float]:
        """Find average c0 of collection of loops by dividing them into
        quartiles by length"""
        quart_loop_df = self._get_quartile_dfs(self._loop_df)
        return list(map(self.find_avg_around_pos, [pos] * 4, [lim] * 4, quart_loop_df))


class MultiChrLoops:
    """
    Abstraction to analyze loops in multiple chromosomes
    """
    def __init__(self, chrs: tuple[ChrId] = ChrIdList):
        self._chrs = chrs

    def save_avg_c0_stat(self):
        mcloop_df = pd.DataFrame({'Chr': self._chrs})
        model_no = 30
        chrs = mcloop_df['Chr'].apply(
            lambda chr_id: Chromosome(chr_id, Prediction(model_no)) 
                if chr_id != 'VL' else Chromosome(chr_id, None)
        )
        mcloop_df['c0'] = chrs.apply(lambda chr: chr.get_spread().mean())
        
        mc_loops = chrs.apply(lambda chr: Loops(chr))
        mcloop_df['loop'] = mc_loops.apply(lambda loops: loops.find_avg_c0())
        
        # Add quartile by length columns
        quart_len_cols = ['quart_len_1', 'quart_len_2', 'quart_len_3', 'quart_len_4']
        mcloop_df[quart_len_cols] = pd.DataFrame(
            np.array(
                mc_loops.apply(
                    lambda loops: loops.find_avg_c0_in_quartile_by_len()
                ).tolist()
            ), 
            columns=quart_len_cols
        )

        # Add quartile by position columns
        quart_pos_cols = ['quart_pos_1', 'quart_pos_2', 'quart_pos_3', 'quart_pos_4']
        mcloop_df[quart_pos_cols] = pd.DataFrame(
            np.array(
                mc_loops.apply(
                    lambda loops: loops.find_avg_c0_in_quartile_by_pos()
                ).tolist()
            ), 
            columns=quart_pos_cols
        )

        # Add Quartile by position in quartile by length columns
        def _make_avg_arr(func: function, *args):
            """Calculate values to append to dataframe by calling functions of
            Loops objects"""
            return np.array(
                mc_loops.apply(
                    lambda loops: func(loops, *args)
                ).tolist()
            )


        quart_len_pos_cols = [f'quart_len_{p[0]}_pos_{p[1]}' 
                        for p in itertools.product(range(1,5), range(1,5))]
        
        mcloop_df[quart_len_pos_cols] = pd.DataFrame(
            _make_avg_arr(Loops.find_avg_c0_in_quartile_by_pos_in_quart_len),
            columns=quart_len_pos_cols
        )
        # mcloop_df[quart_pos_cols] = pd.DataFrame(
        #     np.array(
        #         mc_loops.apply(
        #             lambda loops: loops.find_avg_c0_in_quartile_by_pos_in_quart_len()
        #         ).tolist()
        #     ), 
        #     columns=quart_len_pos_cols
        # )

        # Add anchor and center columns
        anc_bp_it = itertools.product(['start', 'center', 'end'], [500, 200, 50])
        anc_bp_cols = [f'{p[0]}_{p[1]}' for p in anc_bp_it]
        mcloop_df[anc_bp_cols] = pd.concat([ 
            mc_loops.apply(lambda loops: loops.find_avg_around_anc(p[0], p[1]))
            for p in anc_bp_it 
        ], axis=1)

        # Add anchor and center columns of quartile by len 
        quart_anc_len_cols = [f'{p[0]}_200_quart_len_{p[1]}'
                for p in itertools(['start', 'center', 'end'], range(1,5))]
        mcloop_df[quart_anc_len_cols] = pd.concat([pd.DataFrame(
            _make_avg_arr(Loops.find_avg_around_anc_in_quartile_by_len, pos, 200),
            columns=quart_anc_len_cols
        ) for pos in ['start', 'center', 'end']])

        mcloop_df.drop(columns=['c0','Chr']) = mcloop_df.drop(columns=['c0', 'Chr']).apply(lambda col: col - mcloop_df['c0'])
        
        IOUtil().append_tsv(mcloop_df, f'data/generated_data/loop/multichr_avg_c0_stat_{model_no}.tsv')
