from __future__ import annotations
from custom_types import NonNegativeInt

from nucleosome import Nucleosome
from chromosome import Chromosome 
from util import IOUtil, PlotUtil

import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
import numpy as np

from pathlib import Path


class Loops:
    """
    Abstraction of collection of loops in a single chromosome
    """
    COL_MEAN_C0_FULL = 'mean_c0_full'
    COL_MEAN_C0_NUC = 'mean_c0_nuc'
    COL_MEAN_C0_LINKER = 'mean_c0_linker'

    def __init__(self, chrm: Chromosome, mxlen: int = None):
        self._loop_file = f'data/input_data/loops/merged_loops_res_500_chr{chrm._chr_num}.bedpe'
        # TODO: Make _chr, _loop_df public. Used in MeanLoops
        self._chr = chrm
        self._loop_df = self._read_loops()
        if mxlen:
            self.exclude_above_len(mxlen)

    def __len__(self) -> int:
        return len(self._loop_df)

    def __getitem__(self, key: NonNegativeInt) -> pd.Series:
        assert key < len(self)
        return self._loop_df.iloc[key]

    def _read_loops(self) -> pd.DataFrame:
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: [res, start, end, len]
        """
        df = pd.read_table(self._loop_file, skiprows=[1])

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

        return _use_centroids().assign(len=lambda df: df['end'] - df['start'])
    
    def stat_loops(self) -> None:
        """Prints statistics of loops"""
        loop_df = self._loop_df
        max_loop_length = 100000
        loop_df = loop_df.loc[
            loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()
        loop_df = loop_df.assign(length=lambda df: df['end'] - df['start'])
        loop_df['length'].plot(kind='hist')
        plt.xlabel('Loop length (bp)')
        plt.title(
            f'Histogram of loop length. Mean = {loop_df["length"].mean()}bp. Median = {loop_df["length"].median()}bp'
        )

        # TODO: Encapsulate saving figure logic in a function
        fig_dir = 'figures/chrv/loops'
        if not Path(fig_dir).is_dir():
            Path(fig_dir).mkdir(parents=True, exist_ok=True)

        plt.gcf().set_size_inches(12, 6)
        plt.savefig(
            f'{fig_dir}/loop_highres_hist_maxlen_{max_loop_length}.png',
            dpi=200)

    def exclude_above_len(self, mxlen: int) -> pd.DataFrame:
        self._loop_df = self._loop_df.loc[self._loop_df['len'] <= mxlen]

    def get_loop_cover(self, loop_df : pd.DataFrame | None = None) -> np.ndarray:
        if loop_df is None: 
            loop_df = self._loop_df

        loop_array = np.full((self._chr._total_bp, ), False)
        
        def _set_bp(start: float, end: float) -> None:
            loop_array[int(start) : int(end)] = True

        loop_df.apply(lambda loop: _set_bp(loop['start'] - 1, loop['end']), axis=1)
        return loop_array 
    
    def add_mean_c0(self) -> pd.Series:
        """Find mean c0 of full, nucs and linkers of each loop and store it
        
        Returns: 
            A tuple: Name of columns appended to dataframe 
        """
        # TODO: Save columns as class attribute
        mean_cols = [self.COL_MEAN_C0_FULL, self.COL_MEAN_C0_NUC, self.COL_MEAN_C0_LINKER]
        
        if all(list(map(lambda col : col in self._loop_df.columns, mean_cols))):
            return pd.Series(mean_cols)

        c0_spread = self._chr.get_spread()
        nucs = Nucleosome(self._chr)
        nucs_cover = nucs.get_nuc_regions()

        def _mean_of(loop: pd.Series) -> pd.Series:
            """Find mean c0 of full, nucs and linkers of a loop"""
            loop_cover = np.full((self._chr._total_bp,), False)
            loop_cover[loop['start'] - 1 : loop['end']] = True 
            loop_mean = c0_spread[loop_cover].mean()
            loop_nuc_mean = c0_spread[loop_cover & nucs_cover].mean()
            loop_linker_mean = c0_spread[loop_cover & ~nucs_cover].mean()
            return pd.Series([loop_mean, loop_nuc_mean, loop_linker_mean])
        
        self._loop_df[mean_cols] = \
            self._loop_df.apply(_mean_of, axis=1)
        
        return pd.Series(mean_cols)
        
    def plot_c0_in_individual_loop(self):
        loop_df = self._loop_df

        for i in range(len(loop_df)):
            row = loop_df.iloc[i]
            # TODO: -150% to +150% of loop. Vertical line = loop anchor
            self._chr.plot_moving_avg(row['start'], row['end'])
            plt.ylim(-0.7, 0.7)
            plt.xlabel(f'Position along Chromosome {self._chr._chr_num}')
            plt.ylabel('Intrinsic Cyclizability')
            plt.title(
                f'C0 in loop between {row["start"]}-{row["end"]}. Found with resolution: {row["res"]}.'
            )

            IOUtil().save_figure(
                f'figures/loop/{self._chr._chr_id}/{row["start"]}_{row["end"]}.png'
            )

    def _plot_mean_across_loops(self, total_perc: int, chr_spread: np.ndarray,
                                val_type: str) -> None:
        """
        Underlying plotter to plot mean across loops
        
        Plots mean C0 or mean nuc. occupancy. Does not add labels. 
        """
        loop_df = self._loop_df

        # Filter loops by length
        max_loop_length = 100000
        loop_df = loop_df.loc[
            loop_df['end'] - loop_df['start'] < max_loop_length].reset_index()

        def _find_value_in_loop(row: pd.Series) -> np.ndarray:
            """
            Find value from start to end considering total percentage. 
            
            Returns:
                A 1D numpy array. If value can't be calculated for whole total percentage 
                an empty array of size 0 is returned. 
            """
            start_pos = int(row['start'] + (row['end'] - row['start']) *
                            (1 - total_perc / 100) / 2)
            end_pos = int(row['end'] + (row['end'] - row['start']) *
                          (total_perc / 100 - 1) / 2)

            if start_pos < 0 or end_pos > self._chr._total_bp - 1:
                print(f'Excluding loop: ({row["start"]}-{row["end"]})!')
                return np.empty((0, ))

            return chr_spread[start_pos:end_pos]

        assert _find_value_in_loop(pd.Series({
            'start': 30,
            'end': 50
        })).size == int(20 * total_perc / 100)
        assert _find_value_in_loop(pd.Series({
            'start': 50,
            'end': 30
        })).size == 0

        # Find C0 along loop. Resize. Take mean.
        value_in_loops: pd.Series = loop_df.apply(_find_value_in_loop, axis=1)
        value_in_loops = pd.Series(
            list(filter(lambda arr: arr.size != 0, value_in_loops)))

        resize_multiple = 10
        value_in_loops = pd.Series(
            list(
                map(
                    lambda arr: resize(arr,
                                       ((total_perc + 1) * resize_multiple, )),
                    value_in_loops)))
        mean_val = np.array(value_in_loops.tolist()).mean(axis=0)

        plt.close()
        plt.clf()

        # Plot mean value
        x = np.arange(
            (total_perc + 1) *
            resize_multiple) / resize_multiple - (total_perc - 100) / 2
        plt.plot(x, mean_val, color='tab:blue')
        self._chr.plot_horizontal_line(chr_spread.mean())
        plt.grid()

        y_lim = plt.gca().get_ylim()

        # Plot anchor lines
        if total_perc >= 100:
            for pos in [0, 100]:
                plt.axvline(x=pos, color='tab:green', linestyle='--')
                plt.text(pos,
                         y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75,
                         'anchor',
                         color='tab:green',
                         ha='left',
                         va='center')

        # Plot center line
        plt.axvline(x=50, color='tab:orange', linestyle='--')
        plt.text(50,
                 y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75,
                 'center',
                 color='tab:orange',
                 ha='left',
                 va='center')

        # Label plot
        plt.xlabel('Position along loop (percentage)')
        plt.ylabel(val_type)
        plt.title(
            f'Mean {self._chr._c0_type} {val_type} along chromosome {self._chr._chr_num} loop ({x[0]}% to {x[-1]}% of loop length)'
        )

        IOUtil().save_figure(
            f'figures/loop/mean_{val_type}_p_{total_perc}_mxl_{max_loop_length}_{self._chr}.png'
        )
    
    def plot_mean_c0_across_loops(self, total_perc=150) -> None:
        """
        Plot mean C0 across total loop in found loops in chr V

        Args: 
            total_perc: Total percentage of loop length to consider 
        """
        self._plot_mean_across_loops(total_perc, self._chr.get_spread(), 'c0')
    
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
                plt.title(
                    f'C0 around chromosome {self._chr._chr_num} loop {col} anchor at {a}bp. Found with res {loop_df.iloc[i]["res"]}'
                )

                # Save figure
                IOUtil().save_figure(
                    f'figures/loop_anchor/{self._chr}/{col}_{a}.png')

    def plot_c0_around_anchor(self, lim=500):
        """Plot C0 around loop anchor points"""
        # TODO: Distance from loop anchor : percentage. Not required?
        loop_df = self._loop_df

        anchors = np.concatenate(
            (loop_df['start'].to_numpy(), loop_df['end'].to_numpy()))
        
        mean_c0_start = self._chr.mean_c0_around_bps(loop_df['start'], lim, lim)
        mean_c0_end = self._chr.mean_c0_around_bps(loop_df['end'], lim, lim)
        mean_c0_all = self._chr.mean_c0_around_bps(anchors, lim, lim)
        
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
        plt.title(
            f'Mean {self._chr._c0_type} C0 around anchor points. Considering start, end and all anchors.'
        )

        IOUtil().save_figure(f'figures/loop_anchor/dist_{lim}_{self._chr}.png')

    # TODO: Create separate plotter class that take Loop class
    def plot_mean_nuc_occupancy_across_loops(self, total_perc=150) -> None:
        self._plot_mean_across_loops(
            total_perc,
            Nucleosome(self._chr).get_nucleosome_occupancy(), 'nuc_occ')

    def plot_scatter_mean_c0_nuc_linker_individual_loop(self) -> Path:
        nucs = Nucleosome(self._chr)
        nucs_cover = nucs.get_nuc_regions()
        loops_cover = self.get_loop_cover()
        c0_spread = self._chr.get_spread()
        
        mean_cols = self.add_mean_c0()
        sorted_loop_df = self._loop_df.sort_values('len', ignore_index=True)

        # Plot scatter for mean C0 of nuc, linker
        markers = ['o', 's', 'p']
        labels = [ 'loop', 'loop nuc', 'loop linker']
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        plt.close()
        plt.clf()

        PlotUtil().show_grid()

        x = np.arange(len(sorted_loop_df))
        for i, col in enumerate(sorted_loop_df[mean_cols]):
            plt.scatter(x, sorted_loop_df[col], marker=markers[i], label=labels[i], color=colors[i])
        
        # Plot horizontal lines for mean C0 of non-loop nuc, linker 
        non_loop_colors = ['tab:red', 'tab:purple', 'tab:brown']
        non_loops_mean = c0_spread[~loops_cover].mean()
        non_loops_nuc_mean = c0_spread[~loops_cover & nucs_cover].mean()
        non_loops_linker_mean = c0_spread[~loops_cover & ~nucs_cover].mean()
        PlotUtil().plot_horizontal_line(non_loops_mean, non_loop_colors[0], 'non-loop')
        PlotUtil().plot_horizontal_line(non_loops_nuc_mean, non_loop_colors[1], 'non-loop nuc')
        PlotUtil().plot_horizontal_line(non_loops_linker_mean, non_loop_colors[2], 'non-loop linker')

        plt.grid()

        # Decorate
        xticks = sorted_loop_df['len'].apply(lambda len: str(int(len / 1000)) + 'k').tolist()
        plt.xticks(x, xticks, rotation=90)
        plt.xlabel('Individual loops labeled with and sorted by length')
        plt.ylabel('Mean C0')
        plt.title(
            f'Comparison of mean {self._chr._c0_type} C0 among loops'
            f' in chromosome {self._chr._chr_num}'
        )
        plt.legend() 
        
        return IOUtil().save_figure(
            f'figures/loops/individual_scatter_nuc_linker_{self._chr}.png')



