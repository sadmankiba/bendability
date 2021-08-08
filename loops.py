from __future__ import annotations
from nucleosome import Nucleosome
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
    def __init__(self, chrm: Chromosome, mxlen: int = None):
        self._loop_file = f'data/input_data/loops/merged_loops_res_500_chr{chrm._chr_num}.bedpe'
        self._chr = chrm
        self._loop_df = self._read_loops()
        if mxlen:
            self._loop_df = self._exclude_above_len(mxlen)

    def _read_loops(self) -> pd.DataFrame:
        """
        Reads loop positions from .bedpe file

        Returns: 
            A dataframe with three columns: [res, start, end]
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

    def _exclude_above_len(self, mxlen: int) -> pd.DataFrame:
        return self._loop_df.loc[self._loop_df['len'] <= mxlen]

    def get_loop_cover(self, loop_df : pd.DataFrame) -> np.ndarray:
        loop_array = np.full((self._chr._total_bp, ), False)
        
        def _set_bp(start, end) -> None:
            loop_array[start : end] = True

        loop_df.apply(lambda loop: _set_bp(loop['start'] - 1, loop['end']), axis=1)
        return loop_array 

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
        # TODO: Distance from loop anchor : percentage
        loop_df = self._loop_df

        chrv_c0_spread = self._chr.get_spread()

        def mean_around_anchors(anchors: np.ndarray) -> np.ndarray:
            """Calculate mean C0 at bp-resolution around anchors"""
            return np.array(
                list(
                    map(lambda a: chrv_c0_spread[a - 1 - lim:a + lim],
                        anchors))).mean(axis=0)

        anchors = np.concatenate(
            (loop_df['start'].to_numpy(), loop_df['end'].to_numpy()))
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
        plt.title(
            f'Mean {self._chr._c0_type} C0 around anchor points. Considering start, end and all anchors.'
        )

        IOUtil().save_figure(f'figures/loop_anchor/dist_{lim}_{self._chr}.png')

    # TODO: Create separate plotter class that take Loop class
    def plot_mean_nuc_occupancy_across_loops(self, total_perc=150) -> None:
        self._plot_mean_across_loops(
            total_perc,
            Nucleosome(self._chr).get_nucleosome_occupancy(), 'nuc_occ')


class MeanLoops:
    """Class to find mean across loops in various ways"""
    def __init__(self, chrm: Chromosome):
        #TODO: 
        # Takes Loops object as argument
        # Not have _loop_df property 
        self._chrm = chrm
        self._loops = Loops(chrm)
        self._loop_df = self._loops._loop_df

    def _get_quartile_dfs(self, df: pd.DataFrame):
        """Split a dataframe into 4 representing quartile on column len"""
        quart1, quart2, quart3 = df['len'].quantile([0.25, 0.5, 0.75]).tolist()
        return (df.loc[df['len'] <= quart1],
                df.loc[(quart1 < df['len']) & (df['len'] <= quart2)],
                df.loc[(quart2 < df['len']) & (df['len'] <= quart3)],
                df.loc[quart3 < df['len']])

    def in_complete_loop(self, loop_df: pd.DataFrame = None) -> float:
        """
        Find average c0 in loop cover. 

        Args: 
            loop_df: A dataframe of loops. If None, then all loops of this 
                object are considered. 
        """
        if loop_df is None:
            loop_df = self._loop_df

        return round(self._chrm.get_spread()[self._loops.get_loop_cover(loop_df)].mean(), 3)
        
    def in_complete_non_loop(self) -> float:
        return round(self._chrm.get_spread()[~self._loops.get_loop_cover()].mean(), 3)
        
    def in_quartile_by_len(self) -> list[float]:
        """Find average c0 of collection of loops by dividing them into
        quartiles by length"""
        quart_loop_df = self._get_quartile_dfs(self._loop_df)
        return list(map(self.in_complete_loop, quart_loop_df))

    def in_quartile_by_pos(self,
                                       loop_df: pd.DataFrame = None
                                       ) -> np.ndarray:
        """Find average c0 of different positions in collection of loops
        
        Does not use loop cover.

        Returns: 
            A 1D numpy array of size 4
        """
        if loop_df is None:
            loop_df = self._loop_df

        chrv_c0_spread = self._chrm.get_spread()

        def _avg_c0_in_quartile_by_pos(row: pd.Series) -> list[float]:
            quart_pos = row.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).astype(int)
            quart_range = list(
                map(lambda idx: (quart_pos.iloc[idx], quart_pos.iloc[idx + 1]),
                    range(len(quart_pos) - 1)))
            return list(
                map(lambda r: chrv_c0_spread[r[0] - 1:r[1] - 1].mean(),
                    quart_range))

        result = np.round(
            np.mean(loop_df[['start', 'end']].apply(_avg_c0_in_quartile_by_pos,
                                                    axis=1).tolist(),
                    axis=0), 3)
        assert result.shape == (4, )
        return result

    def in_quartile_by_pos_in_quart_len(self) -> np.ndarray:
        """Find average c0 of different positions in quartile of loops by length
        
        Returns: 
            A 1D numpy array of size 16. 
        """
        loop_df = self._loop_df
        quart_loop_df = self._get_quartile_dfs(loop_df)
        return np.array(
            list(map(self.in_quartile_by_pos,
                     quart_loop_df))).flatten()

    def around_anc(self,
                            pos: Literal['start', 'end', 'center'],
                            lim: int = 500,
                            loop_df=None) -> float:
        if loop_df is None:
            loop_df = self._loop_df

        chrv_c0_spread = self._chrm.get_spread()

        loop_df = loop_df.assign(
            center=lambda df: (df['start'] + df['end']) / 2)

        return round(
            sum(
                map(
                    lambda idx: chrv_c0_spread[int(loop_df.iloc[idx][
                        pos]) - lim - 1:int(loop_df.iloc[idx][pos]) + lim].
                    mean(), range(len(loop_df)))) / len(loop_df), 3)

    def around_anc_in_quartile_by_len(self,
                                               pos: Literal['start', 'end',
                                                            'center'],
                                               lim: int = 500) -> list[float]:
        """
        Find average c0 of collection of loops by dividing them into
        quartiles by length
        
        Returns: 
            A list of 4 float numbers
        """
        quart_loop_df = self._get_quartile_dfs(self._loop_df)
        return list(
            map(self.around_anc, [pos] * 4, [lim] * 4, quart_loop_df))

    def in_nuc_linker(self, nuc_half: int = 73) -> tuple[float, float]:
        """
        Returns:  
            A tuple: nuc mean C0, linker mean C0
        """
        nuc_cover = Nucleosome(self._chrm).get_nuc_regions(nuc_half)

        loop_cover = self._loops.get_loop_cover(self._loop_df)
        return (np.round(self._chrm.get_spread()[loop_cover & nuc_cover].mean(), 3), 
            np.round(self._chrm.get_spread()[loop_cover & ~nuc_cover].mean(), 3))
    
    def in_non_loop_nuc_linker(self, nuc_half: int = 73):
        nuc_cover = Nucleosome(self._chrm).get_nuc_regions(nuc_half)

        loop_cover = self._loops.get_loop_cover(self._loop_df)
        return (np.round(self._chrm.get_spread()[~loop_cover & nuc_cover].mean(), 3), 
            np.round(self._chrm.get_spread()[~loop_cover & ~nuc_cover].mean(), 3))


class MultiChrmMeanLoopsCollector:
    """
    Class to accumulate various mean functions in loops in a dataframe for
    side-by-side comparison. 
    """
    def __init__(self, prediction : Prediction, chrids: tuple[ChrId] = ChrIdList):
        # TODO: Rename collector_df
        self._mcloop_df = pd.DataFrame({'ChrID': chrids})  
        self._prediction = prediction
        self._chrs = self._get_chromosomes()
        self._mcloops = self._chrs.apply(lambda chrm: MeanLoops(chrm))
        self._mcnucs = self._chrs.apply(lambda chrm: Nucleosome(chrm))

    def _get_chromosomes(self) -> pd.Series:
        """Create a Pandas Series of Chromosomes"""
        return self._mcloop_df['ChrID'].apply(
            lambda chr_id: Chromosome(chr_id, self._prediction)
            if chr_id != 'VL' else Chromosome(chr_id, None))

    def _create_multiple_col(self, func: function, *args) -> np.ndarray:
        """
        Call functions of Loops in each chromosome and split result into
        multiple columns

        Returns: A 2D numpy array, where each column represents a column for
            dataframe
        """
        return np.array(
            self._mcloops.apply(lambda mloops: func(mloops, *args)).tolist())

    def _add_chrm_mean(self) -> None:
        self._mcloop_df['chromosome'] = self._chrs.apply(
            lambda chr: chr.get_spread().mean())

    def _add_chrm_nuc_linker_mean(self) -> None:
        nuc_linker_arr = np.array(
            self._mcnucs.apply(
                lambda nucs: nucs.find_avg_nuc_linker_c0()).tolist())
        nuc_linker_cols = ['chrm_nuc', 'chrm_linker']
        self._mcloop_df[nuc_linker_cols] = pd.DataFrame(
            nuc_linker_arr, columns=nuc_linker_cols)

    def _add_loop_cover_frac(self) -> None:
        self._mcloop_df['cover'] = self._mcloops.apply(
            lambda mloops: mloops._loops.get_loop_cover().mean())

    def _add_loop_mean(self) -> None:
        self._mcloop_df['loop'] = self._mcloops.apply(
            lambda mloops: mloops.in_complete_loop())

    def _add_loop_nuc_linker_mean(self) -> None:
        loop_nuc_linker_cols = ['loop_nuc', 'loop_linker']
        self._mcloop_df[loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_nuc_linker),
            columns=loop_nuc_linker_cols)

    def _add_non_loop_mean(self) -> None:
        self._mcloop_df['non_loop'] = self._mcloops.apply(
            lambda mloops: mloops.in_complete_non_loop())
    
    def _add_non_loop_nuc_linker_mean(self) -> None: 
        non_loop_nuc_linker_cols = ['non_loop_nuc', 'non_loop_linker']
        self._mcloop_df[non_loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_non_loop_nuc_linker),
            columns=non_loop_nuc_linker_cols)

    def _add_quartile_by_len(self) -> None:
        quart_len_cols = [
            'quart_len_1', 'quart_len_2', 'quart_len_3', 'quart_len_4'
        ]
        self._mcloop_df[quart_len_cols] = pd.DataFrame(self._create_multiple_col(
            MeanLoops.in_quartile_by_len),
                                                       columns=quart_len_cols)

    def _add_quartile_by_pos(self) -> None:
        quart_pos_cols = [
            'quart_pos_1', 'quart_pos_2', 'quart_pos_3', 'quart_pos_4'
        ]
        self._mcloop_df[quart_pos_cols] = pd.DataFrame(self._create_multiple_col(
            MeanLoops.in_quartile_by_pos),
                                                       columns=quart_pos_cols)

    def _add_quartile_by_len_pos(self) -> None:
        quart_len_pos_cols = [
            f'quart_len_{p[0]}_pos_{p[1]}'
            for p in itertools.product(range(1, 5), range(1, 5))
        ]
        self._mcloop_df[quart_len_pos_cols] = pd.DataFrame(
            self._create_multiple_col(
                MeanLoops.in_quartile_by_pos_in_quart_len),
            columns=quart_len_pos_cols)

    def _add_anchor_center_bp(self) -> None:
        def _get_anc_bp_it():
            return itertools.product(['start', 'center', 'end'],
                                     [500, 200, 50])

        anc_bp_cols = [f'{p[0]}_{p[1]}' for p in _get_anc_bp_it()]

        anc_bp_cols_df = pd.concat([
            pd.Series(self._mcloops.apply(
                lambda mloops: mloops.around_anc(p[0], p[1])),
                      name=f'{p[0]}_{p[1]}') for p in _get_anc_bp_it()
        ],
                                   axis=1)
        self._mcloop_df[anc_bp_cols] = anc_bp_cols_df

    def _add_quartile_len_anchor_center_bp(self) -> None:
        quart_anc_len_cols = [
            f'{p[0]}_200_quart_len_{p[1]}'
            for p in itertools.product(['start', 'center', 'end'], range(1, 5))
        ]
        self._mcloop_df[quart_anc_len_cols] = pd.concat([
            pd.DataFrame(self._create_multiple_col(
                MeanLoops.around_anc_in_quartile_by_len, pos, 200),
                         columns=quart_anc_len_cols[pos_idx * 4:(pos_idx + 1) *
                                                    4])
            for pos_idx, pos in enumerate(['start', 'center', 'end'])
        ],
                                                        axis=1)

    def _subtract_mean_chrm_c0(self) -> None:
        exclude_cols = ['chromosome','ChrID', 'cover']
        self._mcloop_df[self._mcloop_df.drop(columns=exclude_cols).columns] = \
            self._mcloop_df.drop(columns=exclude_cols).apply(lambda col: col - self._mcloop_df['chromosome'])

    def save_avg_c0_stat(self,
                         mean_methods: list[int],
                         subtract_chrm=True) -> None:
        """
        Args:
            mean_methods: A list of int. Between 0-6 (inclusive)
            subtract_chrm: Whether to subtract chrm c0
        """
        method_map = {
            0: self._add_chrm_mean,
            1: self._add_chrm_nuc_linker_mean,
            2: self._add_loop_cover_frac,
            3: self._add_loop_mean,
            4: self._add_loop_nuc_linker_mean,
            5: self._add_non_loop_mean,
            6: self._add_non_loop_nuc_linker_mean,
            7: self._add_quartile_by_len,
            8: self._add_quartile_by_pos,
            9: self._add_quartile_by_len_pos,
            10: self._add_anchor_center_bp,
            11: self._add_quartile_len_anchor_center_bp,
        }

        for m in mean_methods:
            method_map[m]()

        if subtract_chrm:
            self._subtract_mean_chrm_c0()

        self._mcloop_df['model'] = np.full((len(self._mcloop_df), ),
                                           str(self._prediction))

        IOUtil().append_tsv(
            self._mcloop_df,
            f'data/generated_data/loop/multichr_avg_c0_stat_m_{self._prediction}.tsv'
        )

    def plot_loop_nuc_linker_mean(self):
        self.save_avg_c0_stat([2, 3, 4, 5], subtract_chrm=False)
        labels = [ 'loop', 'loop_nuc', 'loop_linker', 
            'non_loop', 'non_loop_nuc', 'non_loop_linker']

        arr = self._mcloop_df[labels].values
        x = np.arange(arr.shape[0])
        markers = ['o', 's', 'p', 'P', '*', 'D']
        for i in range(arr.shape[1]):
            plt.scatter(x, arr[:, i], marker=markers[i], label=labels[i])

        plt.xticks(x, self._mcloop_df['ChrID'])
        plt.xlabel('Chromosome')
        plt.ylabel('Mean C0')
        plt.title(
            'Comparison of mean C0 in nucleosome and linker region in loops vs. total chromosome'
        )
        plt.legend()

        IOUtil().save_figure(
            f'figures/mcloop/nuc_linker_mean_{self._prediction}.png')
