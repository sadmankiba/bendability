from __future__ import annotations
from nucleosome import Nucleosome
from typing import IO

from custom_types import ChrId
from prediction import Prediction
from reader import DNASequenceReader
from constants import ChrIdList
from chromosome import Chromosome 
from util import IOUtil, PlotUtil

import matplotlib.pyplot as plt
import matplotlib as mpl
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
        # TODO: Make _chr, _loop_df public. Used in MeanLoops
        self._chr = chrm
        self._loop_df = self._read_loops()
        if mxlen:
            self.exclude_above_len(mxlen)

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
        mean_cols = ['mean_c0_full', 'mean_c0_nuc', 'mean_c0_linker']
        
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



class MeanLoops:
    """Class to find mean across loops in various ways"""
    def __init__(self, loops: Loops):
        self._loops = loops
        
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
            loop_df = self._loops._loop_df

        return round(self._loops._chr.get_spread()[self._loops.get_loop_cover(loop_df)].mean(), 3)
        
    def in_complete_non_loop(self) -> float:
        return round(self._loops._chr.get_spread()[
            ~self._loops.get_loop_cover(self._loops._loop_df)].mean(), 3)
        
    def in_quartile_by_len(self) -> list[float]:
        """Find average c0 of collection of loops by dividing them into
        quartiles by length"""
        quart_loop_df = self._get_quartile_dfs(self._loops._loop_df)
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
            loop_df = self._loops._loop_df

        chrv_c0_spread = self._loops._chr.get_spread()

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
        loop_df = self._loops._loop_df
        quart_loop_df = self._get_quartile_dfs(loop_df)
        return np.array(
            list(map(self.in_quartile_by_pos,
                     quart_loop_df))).flatten()

    def around_anc(self,
                            pos: Literal['start', 'end', 'center'],
                            lim: int = 500,
                            loop_df=None) -> float:
        if loop_df is None:
            loop_df = self._loops._loop_df

        chrv_c0_spread = self._loops._chr.get_spread()

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
        quart_loop_df = self._get_quartile_dfs(self._loops._loop_df)
        return list(
            map(self.around_anc, [pos] * 4, [lim] * 4, quart_loop_df))

    def in_nuc_linker(self, nuc_half: int = 73) -> tuple[float, float]:
        """
        Returns:  
            A tuple: nuc mean C0, linker mean C0
        """
        nuc_cover = Nucleosome(self._loops._chr).get_nuc_regions(nuc_half)

        loop_cover = self._loops.get_loop_cover(self._loops._loop_df)
        return (np.round(self._loops._chr.get_spread()[loop_cover & nuc_cover].mean(), 3), 
            np.round(self._loops._chr.get_spread()[loop_cover & ~nuc_cover].mean(), 3))
    
    def in_non_loop_nuc_linker(self, nuc_half: int = 73):
        nuc_cover = Nucleosome(self._loops._chr).get_nuc_regions(nuc_half)

        loop_cover = self._loops.get_loop_cover(self._loops._loop_df)
        return (np.round(self._loops._chr.get_spread()[~loop_cover & nuc_cover].mean(), 3), 
            np.round(self._loops._chr.get_spread()[~loop_cover & ~nuc_cover].mean(), 3))


class MultiChrmMeanLoopsCollector:
    """
    Class to accumulate various mean functions in loops in a dataframe for
    side-by-side comparison. 
    """
    def __init__(self, 
                prediction : Prediction, 
                chrids: tuple[ChrId] = ChrIdList, 
                mxlen: int | None = None):
        # TODO: Rename collector_df
        self._mcloop_df = pd.DataFrame({'ChrID': chrids})  
        self._prediction = prediction
        self._chrs = self._get_chromosomes()
        
        mcloops = self._chrs.apply(lambda chrm: Loops(chrm, mxlen))
        self._mcmloops = mcloops.apply(lambda loops: MeanLoops(loops))
        self._mcnucs = self._chrs.apply(lambda chrm: Nucleosome(chrm))
        self._mxlen = mxlen
    
    def __str__(self):
        ext = 'with_vl' if 'VL' in self._mcloop_df['ChrID'].tolist() \
                else 'without_vl'
        
        return f'md_{self._prediction}_mx_{self._mxlen}_{ext}'

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
            self._mcmloops.apply(lambda mloops: func(mloops, *args)).tolist())

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
        self._mcloop_df['cover'] = self._mcmloops.apply(
            lambda mloops: mloops._loops.get_loop_cover().mean())

    def _add_loop_mean(self) -> None:
        self._mcloop_df['loop'] = self._mcmloops.apply(
            lambda mloops: mloops.in_complete_loop())

    def _add_loop_nuc_linker_mean(self) -> None:
        loop_nuc_linker_cols = ['loop_nuc', 'loop_linker']
        self._mcloop_df[loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_nuc_linker),
            columns=loop_nuc_linker_cols)

    def _add_non_loop_mean(self) -> Literal['non_loop']:
        if 'non_loop' not in self._mcloop_df.columns: 
            self._mcloop_df['non_loop'] = self._mcmloops.apply(
                lambda mloops: mloops.in_complete_non_loop())
        
        return 'non_loop' 
    
    def _add_non_loop_nuc_linker_mean(self) -> None: 
        non_loop_nuc_linker_cols = ['non_loop_nuc', 'non_loop_linker']
        self._mcloop_df[non_loop_nuc_linker_cols] = pd.DataFrame(
            self._create_multiple_col(MeanLoops.in_non_loop_nuc_linker),
            columns=non_loop_nuc_linker_cols)
        return non_loop_nuc_linker_cols

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
            pd.Series(self._mcmloops.apply(
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
        exclude_cols = ['chromosome','ChrID']
        if 'cover' in self._mcloop_df.columns:
            exclude_cols += ['cover']
        self._mcloop_df[self._mcloop_df.drop(columns=exclude_cols).columns] = \
            self._mcloop_df.drop(columns=exclude_cols).apply(lambda col: col - self._mcloop_df['chromosome'])

    def _add_num_loops(self) -> Literal['num_loops']:
        # TODO: Use wrapper hof, check cols 
        self._mcloop_df['num_loops'] = self._mcmloops.apply(
            lambda mloops: len(mloops._loops._loop_df))
        
        return 'num_loops'
    
    def _add_num_loops_lt_non_loop(self) -> Literal['num_loops_lt_nl']:
        # TODO: Use func id and col_for 
        num_loops_lt_nl_col = 'num_loops_lt_nl'
        
        if num_loops_lt_nl_col in self._mcloop_df.columns:
            return num_loops_lt_nl_col
        
        # Add mean c0 in each loop in loops object 
        mean_cols = self._mcmloops.apply(lambda mloops: mloops._loops.add_mean_c0()).iloc[0]
        
        # Add non loop mean of each chromosome to this object 
        non_loop_col = self._add_non_loop_mean()

        # Compare mean c0 of whole loop and non loop
        mcmloops = pd.Series(self._mcmloops, name='mcmloops')
        self._mcloop_df[num_loops_lt_nl_col] = pd.DataFrame(mcmloops).apply(
            lambda mloops: (mloops['mcmloops']._loops._loop_df[mean_cols[0]] 
                < self._mcloop_df.iloc[mloops.name][non_loop_col]).sum(),
            axis=1
        )

        return num_loops_lt_nl_col

    def _add_num_loops_l_lt_nll(self) -> Literal['num_loops_l_lt_nll']:
        func_id = 14
        if self.col_for(func_id) in self._mcloop_df.columns:
            return self.col_for(func_id)
        
        # Add mean c0 in each loop in loops object 
        mean_cols = self._mcmloops.apply(lambda mloops: mloops._loops.add_mean_c0()).iloc[0]
        
        # Add non loop linker mean of each chromosome to this object 
        _, nl_l_col = self._add_non_loop_nuc_linker_mean()

        # Compare mean c0 of linkers in loop and non loop linkers
        mcmloops = pd.Series(self._mcmloops, name='mcmloops')
        self._mcloop_df[self.col_for(func_id)] = pd.DataFrame(mcmloops).apply(
            lambda mloops: (mloops['mcmloops']._loops._loop_df[mean_cols[2]] 
                < self._mcloop_df.iloc[mloops.name][nl_l_col]).sum(),
            axis=1
        )

        return self.col_for(func_id)

    def col_for(self, func_id: int):
        col_map = {
            12: 'num_loops',
            13: 'num_loops_lt_nl', 
            14: 'num_loops_l_lt_nll'
        }
        return col_map[func_id]

    def save_avg_c0_stat(self,
                         mean_methods: list[int] | None = None,
                         subtract_chrm=True) -> str:
        """
        Args:
            mean_methods: A list of int. Between 0-11 (inclusive)
            subtract_chrm: Whether to subtract chrm c0
        
        Returns:
            The path where dataframe is saved
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
            12: self._add_num_loops,
            13: self._add_num_loops_lt_non_loop,
            14: self._add_num_loops_l_lt_nll
        }

        # Select all
        if mean_methods is None:
            mean_methods = list(method_map.keys())

        for m in mean_methods:
            method_map[m]()

        if subtract_chrm:
            self._subtract_mean_chrm_c0()

        self._mcloop_df['model'] = np.full((len(self._mcloop_df), ),
                                           str(self._prediction))

        save_df_path = f'data/generated_data/mcloops/multichr_avg_c0_stat_{self}.tsv'
        IOUtil().append_tsv(
            self._mcloop_df,
            save_df_path
        )
        return save_df_path

    def plot_scatter_loop_nuc_linker_mean(self):
        self.save_avg_c0_stat([3, 4, 5, 6], subtract_chrm=False)
        labels = [ 'loop', 'loop_nuc', 'loop_linker', 
            'non_loop', 'non_loop_nuc', 'non_loop_linker']

        # Show grid below other plots
        plt.rc('axes', axisbelow=True)

        arr = self._mcloop_df[labels].values
        x = np.arange(arr.shape[0])
        markers = ['o', 's', 'p', 'P', '*', 'D']
        for i in range(arr.shape[1]):
            plt.scatter(x, arr[:, i], marker=markers[i], label=labels[i])

        plt.grid()
        plt.xticks(x, self._mcloop_df['ChrID'])
        plt.xlabel('Chromosome')
        plt.ylabel('Mean C0')
        plt.title(
            'Comparison of mean C0 in nucleosome and linker region in loop'
            f' vs. non-loop with max loop length = {self._mxlen}'
        )
        plt.legend()

        IOUtil().save_figure(
            f'figures/mcloop/nuc_linker_mean_{self}.png')
    
    def plot_loop_cover_frac(self):
        self.save_avg_c0_stat([2], subtract_chrm=False)
        cv_frac = self._mcloop_df['cover'].to_numpy() * 100
        x = np.arange(cv_frac.size)
        plt.bar(x, cv_frac)
        plt.grid()
        plt.xticks(x, self._mcloop_df['ChrID'])
        plt.xlabel('Chromosome')
        plt.ylabel('Loop cover (%)')
        plt.title(f'Loop cover percentage in whole chromosome with max length = {self._mxlen}')
        IOUtil().save_figure(
            f'figures/mcloop/loop_cover_{self}.png')


class MultiChrmMeanLoopsAggregator:
    def __init__(self, coll: MultiChrmMeanLoopsCollector):
        self._coll = coll
        self._agg_df = pd.DataFrame({'ChrIDs': [coll._mcloop_df['ChrID'].tolist()] })

    def _loop_lt_nl(self):
        self._coll.save_avg_c0_stat([12, 13], False)
        lp_lt_nl = self._coll._mcloop_df[self._coll.col_for(13)].sum() \
            / self._coll._mcloop_df[self._coll.col_for(12)].sum()
        self._agg_df['loop_lt_nl'] = lp_lt_nl * 100
    
    def _loop_l_lt_nll(self):
        self._coll.save_avg_c0_stat([12, 14], False)
        self._agg_df['loop_l_lt_nll'] = self._coll._mcloop_df[self._coll.col_for(14)].sum() \
            / self._coll._mcloop_df[self._coll.col_for(12)].sum() * 100

    def save_stat(self, methods : list[int]) -> Path:
        method_map = {
            0: self._loop_lt_nl,
            1: self._loop_l_lt_nll
        }

        for m in methods:
            method_map[m]()

        save_df_path = f'data/generated_data/mcloops/agg_stat_{self._coll}.tsv'
        return IOUtil().append_tsv(self._agg_df, save_df_path)
        

class CoverLoops:
    def __init__(self, loops: Loops):
        nucs = Nucleosome(loops._chr)
        
        self._nuc_cover = nucs.get_nuc_regions()
        self._loop_cover = loops.get_loop_cover(loops._loop_df)

    def in_loop_nuc(self) -> float:
        return (self._loop_cover & self._nuc_cover).mean()
    
    def in_loop_linker(self) -> float: 
        return (self._loop_cover & ~self._nuc_cover).mean()
    
    def in_non_loop_nuc(self) -> float:
        return (~self._loop_cover & self._nuc_cover).mean()
    
    def in_non_loop_linker(self) -> float:
        return (~self._loop_cover & ~self._nuc_cover).mean()


class MultiChrmCoverLoopsCollector:
    def __init__(self, chrmids: tuple[ChrId] = ChrIdList, mxlen: int | None = None):
        self._chrmids = chrmids
        self._mxlen = mxlen

        chrms = pd.Series(list(map(lambda chrm_id: Chromosome(chrm_id), chrmids)))
        mcloops = chrms.apply(lambda chrm: Loops(chrm, mxlen))
        self._mccloops = mcloops.apply(lambda loops: CoverLoops(loops))
    
    def get_cover_stat(self) -> pd.DataFrame:
        collector_df = pd.DataFrame({'ChrID': self._chrmids})
        
        collector_df['loop_nuc'] = self._mccloops.apply(
            lambda cloops: cloops.in_loop_nuc()
        )

        collector_df['loop_linker'] = self._mccloops.apply(
            lambda cloops: cloops.in_loop_linker()
        )

        collector_df['non_loop_nuc'] = self._mccloops.apply(
            lambda cloops: cloops.in_non_loop_nuc()
        )

        collector_df['non_loop_linker'] = self._mccloops.apply(
            lambda cloops: cloops.in_non_loop_linker()
        )
        save_path_str = f'data/generated_data/mcloop/multichr_cover_stat_{self._mxlen}.tsv'
        IOUtil().append_tsv(collector_df, save_path_str)
        return collector_df, save_path_str
    
    def plot_bar_cover_stat(self) -> str:
        labels = ['loop_nuc', 'loop_linker', 'non_loop_nuc', 'non_loop_linker']
        colt_df = self.get_cover_stat()[0]
        colt_arr = colt_df[labels].values
        mpl.rcParams.update({'font.size': 12})
        PlotUtil().plot_stacked_bar(colt_arr.transpose() * 100, labels, 
            colt_df['ChrID'].tolist(), show_values=True, value_format="{:.1f}", 
            y_label='Coverage (%)')
        
        plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=4, fancybox=False, shadow=False)

        plt.xlabel('Chromosome')
        plt.title('Coverage by nucleosomes and linkers in loop and' 
            f'non-loop region with max loop length = {self._mxlen}',
            pad=35)
        fig_path_str = f'figures/mcloop/nuc_linker_cover_mxl_{self._mxlen}.png'

        IOUtil().save_figure(fig_path_str)
        return fig_path_str