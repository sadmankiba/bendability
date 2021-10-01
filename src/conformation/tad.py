from __future__ import annotations
from chromosome.genes import Genes

from util.util import IOUtil, PlotUtil, PathUtil 
from util.custom_types import ChrId, PositiveInt, YeastChrNum
from chromosome.chromosome import Chromosome
from models.prediction import Prediction
from util.constants import ChrIdList

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal

class HicExplBoundaries:
    """
    Represenation of boundaries in a chromosome

    Note: Here domains are also determined by boundaries
    """
    def __init__(self, chrm: Chromosome, res: PositiveInt = 500, lim: PositiveInt = 250):
        """
        Args:
            lim: Limit around boundary middle bp to include in boundary
        """
        self._chrm = chrm
        self._res = res
        self.bndrs_df = self._read_boundaries()
        self._lim = lim 
        self._add_mean_c0_col()
        self._add_in_promoter_col()
    
    def __str__(self):
        return f'res_{self._res}_lim_{self._lim}_{self._chrm}'
    
    def _read_boundaries(self) -> pd.DataFrame:
        return pd.read_table(f'{PathUtil.get_data_dir()}/input_data/domains/'
            f'{self._chrm._chr_num}_res_{self._res}_hicexpl_boundaries.bed',
            delim_whitespace=True,
            header=None,
            names=['chromosome', 'left', 'right', 'id', 'score', '_'])\
            .assign(middle=lambda df: (df['left'] + df['right']) // 2)\
                .drop(columns='_')

    def get_domains(self) -> pd.DataFrame:
        dmns_df = pd.DataFrame({'start': self.bndrs_df['right'].tolist()[:-1], 
            'end': self.bndrs_df['left'].tolist()[1:]})
        return dmns_df.assign(len=lambda df: df['end'] - df['start'])

    def _add_mean_c0_col(self) -> None:
        """Add mean c0 of each bndry"""
        self.bndrs_df = self.bndrs_df.assign(mean_c0=lambda df:
            self._chrm.mean_c0_at_bps(df['middle'], self._lim, self._lim))

    def _add_in_promoter_col(self) -> None:
        self.bndrs_df = self.bndrs_df.assign(in_promoter=lambda df: 
                Genes(self._chrm).in_promoter(df['middle']))

    # TODO: Remove bndry, prmtr, non-prmtr bndrs mean c0 method
    # TODO: domain_mean_c0 property?
    def bndry_domain_mean_c0(self) -> tuple[float, float]:
        """
        Returns:
            A tuple: bndry cvr mean, domain cvr mean
        """
        if not(hasattr(self, '_bndrs_mean_c0') or hasattr(self, '_dmns_mean_c0')):
            c0_spread = self._chrm.get_spread()
            bndry_cvr = self._chrm.get_cvr_mask(
                self.bndrs_df['middle'], self._lim, self._lim)
            self._bndrs_mean_c0, self._dmns_mean_c0 = \
                c0_spread[bndry_cvr].mean(), c0_spread[~bndry_cvr].mean()
        
        return self._bndrs_mean_c0, self._dmns_mean_c0 

    def prmtr_bndrs_mean_c0(self) -> float:
        return self._chrm.mean_c0_of_segments(
            self.bndrs_df.iloc[self.bndrs_df['in_promoter'].to_numpy()]
                ['middle'], self._lim, self._lim)
        
    def non_prmtr_bndrs_mean_c0(self) -> float: 
        return self._chrm.mean_c0_of_segments(
            self.bndrs_df.iloc[~(self.bndrs_df['in_promoter'].to_numpy())]['middle'], self._lim, self._lim)

    def num_bndry_mean_c0_greater_than_dmn(self) -> PositiveInt:
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df['mean_c0'] > dmns_mean_c0).sum()

    def num_prmtr_bndry_mean_c0_greater_than_dmn(self) -> float:
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df.query('in_promoter')['mean_c0'] > dmns_mean_c0).sum()
    
    def num_non_prmtr_bndry_mean_c0_greater_than_dmns(self) -> float:
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df.query('not in_promoter')['mean_c0'] > dmns_mean_c0).sum()

    def plot_scatter_mean_c0_each_bndry(self) -> Path:
        markers = ['o', 's']
        labels = [ 'promoter', 'non-promoter']
        colors = ['tab:blue', 'tab:orange']
        
        plt.close()
        plt.clf()

        PlotUtil().show_grid()

        p_x = self.bndrs_df.query('in_promoter')['middle']
        np_x = self.bndrs_df.query('not in_promoter')['middle']
        plt.scatter(p_x, self.bndrs_df.query('in_promoter')['mean_c0'], marker=markers[0], label=labels[0], color=colors[0])
        plt.scatter(np_x, self.bndrs_df.query('not in_promoter')['mean_c0'], marker=markers[1], label=labels[1], color=colors[1])
        
        # Plot horizontal lines for mean C0 of non-loop nuc, linker 
        horiz_colors = ['tab:green', 'tab:red', 'tab:purple']
        bndrs_mean_c0, dmns_mean_c0 = self.bndry_domain_mean_c0()
        chrm_mean_c0 = self._chrm.get_spread().mean()
        PlotUtil().plot_horizontal_line(dmns_mean_c0, horiz_colors[0], 'domains')
        PlotUtil().plot_horizontal_line(chrm_mean_c0, horiz_colors[1], 'chromosome')
        PlotUtil().plot_horizontal_line(bndrs_mean_c0, horiz_colors[2], 'boundaries')

        # Decorate
        plt.xlabel('Position along chromosome (bp)')
        plt.ylabel('Mean C0')
        plt.title(
            f'Comparison of mean {self._chrm._c0_type} C0 among boundaries'
            f' in chromosome {self._chrm._chr_num}'
        )
        plt.legend() 
        
        return IOUtil().save_figure(
            f'{PathUtil.get_figure_dir()}/domains/mean_c0_scatter_{self}.png')



# TODO: 
# - Create common MultiChrm Class for loops and boundaries. Factory? Composition?
# - Have smaller name
class MultiChrmHicExplBoundariesCollector:
    def __init__(self, 
                prediction: Prediction, 
                chrids: tuple[ChrId] = ChrIdList, 
                res: PositiveInt = 500,
                lim: PositiveInt = 250):
        self._prediction = prediction
        self._chrids = chrids
        self._res = res
        self._lim = lim
        self._coll_df = pd.DataFrame({'ChrID': chrids})
        self._chrms = self._get_chrms()
        self._mc_bndrs = self._get_mc_bndrs()

    def __str__(self):
        ext = 'with_vl' if 'VL' in self._chrids else 'without_vl'
        return f'res_{self._res}_lim_{self._lim}_md_{self._prediction}_{ext}'

    def _get_chrms(self) -> pd.Series:
        return pd.Series(list(map(
            lambda chrm_id: Chromosome(chrm_id, self._prediction), self._chrids)))

    def _get_mc_bndrs(self) -> pd.Series[HicExplBoundaries]:
        return self._chrms.apply(lambda chrm: HicExplBoundaries(chrm, self._res, self._lim))

    def _add_num_bndrs(self) -> Literal['num_bndrs']:
        num_bndrs_col = 'num_bndrs'
        if not num_bndrs_col in self._coll_df.columns:
            self._coll_df[num_bndrs_col] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df))
        
        return num_bndrs_col
    
    def _add_num_dmns(self) -> str: 
        num_dmns_col = 'num_dmns'
        if not num_dmns_col in self._coll_df.columns:
            num_bndrs_col = self._add_num_bndrs()
            self._coll_df[num_dmns_col] = self._coll_df[num_bndrs_col] - 1

        return num_dmns_col
    
    def _add_bndrs_mean(self) -> Literal['c0_bndrs']:
        c0_bndrs_col = 'c0_bndrs'
        if not c0_bndrs_col in self._coll_df.columns:
            self._coll_df['c0_bndrs'] = self._mc_bndrs.apply(lambda bndrs: bndrs.bndry_domain_mean_c0()[0])
        
        return c0_bndrs_col
    
    def _add_dmns_mean(self) -> Literal['c0_dmns']:
        c0_dmns_col = 'c0_dmns'
        if not c0_dmns_col in self._coll_df.columns:
            self._coll_df[c0_dmns_col] = self._mc_bndrs.apply(lambda bndrs: bndrs.bndry_domain_mean_c0()[1])
        
        return c0_dmns_col

    def _add_num_bndrs_gt_dmns(self) -> str:
        num_bndrs_gt_dmns_col = 'num_bndrs_gt_dmns'
        if not num_bndrs_gt_dmns_col in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name='mc_bndrs')
            self._coll_df[num_bndrs_gt_dmns_col] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (bndrs['mc_bndrs'].bndrs_df['mean_c0']
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]).sum(),
                axis=1
            )
        
        return num_bndrs_gt_dmns_col
    
    def _add_num_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 5
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name='mc_bndrs')
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (bndrs['mc_bndrs'].bndrs_df.query('in_promoter')['mean_c0']
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]).sum(),
                axis=1
            )
        
        return self.col_for(method_id)
    
    def _add_num_non_prmtr_bndrs_gt_dmns(self) -> str:
        method_id = 6
        if not self.col_for(method_id) in self._coll_df.columns:
            c0_dmns_col = self._add_dmns_mean()

            # Compare mean C0 of each prmtr bndry and dmns
            mc_bndrs = pd.Series(self._mc_bndrs, name='mc_bndrs')
            self._coll_df[self.col_for(method_id)] = pd.DataFrame(mc_bndrs).apply(
                lambda bndrs: (bndrs['mc_bndrs'].bndrs_df.query('not in_promoter')['mean_c0']
                    > self._coll_df.iloc[bndrs.name][c0_dmns_col]).sum(),
                axis=1
            )
        
        return self.col_for(method_id)

    def _add_num_prmtr_bndrs(self) -> Literal['num_p_b']:
        method_id = 7
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query('in_promoter')))
        
        return self.col_for(method_id)
    
    def _add_num_non_prmtr_bndrs(self) -> Literal['num_np_b']:
        method_id = 8
        if not self.col_for(method_id) in self._coll_df.columns:
            self._coll_df[self.col_for(method_id)] = self._mc_bndrs.apply(
                lambda mc_bndrs: len(mc_bndrs.bndrs_df.query('not in_promoter')))
        
        return self.col_for(method_id)

    def col_for(self, method_id: int) -> str:
        col_map = {
            0: 'num_bndrs',
            1: 'num_dmns', 
            2: 'c0_bndrs',
            3: 'c0_dmns',
            4: 'num_bndrs_gt_dmns',
            5: 'num_p_b_gt_d',
            6: 'num_np_b_gt_d',
            7: 'num_p_b',
            8: 'num_np_b'
        }
        return col_map[method_id]

    def save_stat(self, methods: list[int] = None) -> Path:
        method_map = {
            0: self._add_num_bndrs,
            1: self._add_num_dmns,
            2: self._add_bndrs_mean,
            3: self._add_dmns_mean,
            4: self._add_num_bndrs_gt_dmns, 
            5: self._add_num_prmtr_bndrs_gt_dmns,
            6: self._add_num_non_prmtr_bndrs_gt_dmns, 
            7: self._add_num_prmtr_bndrs,
            8: self._add_num_non_prmtr_bndrs
        }
        for m in methods:
            method_map[m]()

        self._coll_df['res'] = np.full((len(self._coll_df), ), self._res)
        self._coll_df['lim'] = np.full((len(self._coll_df), ), self._lim)
        self._coll_df['model'] = np.full((len(self._coll_df), ), str(self._prediction))

        return IOUtil().append_tsv(self._coll_df,
            f'{PathUtil.get_data_dir()}/generated_data/mcdomains/mcdmns_stat.tsv'
        )

    def plot_scatter_mean_c0(self) -> Path:
        """Draw scatter plot of mean c0 at boundaries and domains of
        chromosomes"""
        chrms = self._chrms
        chrm_means = chrms.apply(lambda chrm: chrm.get_spread().mean())
        
        mc_bndrs = self._mc_bndrs
        mc_prmtr_bndrs_c0 = mc_bndrs.apply(lambda bndrs: bndrs.prmtr_bndrs_mean_c0())
        mc_non_prmtr_bndrs_c0 = mc_bndrs.apply(
            lambda bndrs: bndrs.non_prmtr_bndrs_mean_c0())

        mc_bndrs_dmns_c0 = mc_bndrs.apply(lambda bndrs: bndrs.bndry_domain_mean_c0())
        mc_bndrs_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:,0]
        mc_dmns_c0 = np.array(mc_bndrs_dmns_c0.tolist())[:,1]

        # Print comparison
        print('bndrs > dmns:', (mc_bndrs_c0 > mc_dmns_c0).sum())
        print('prmtr bndrs > dmns:', (mc_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print('non prmtr bndrs > dmns:', (mc_non_prmtr_bndrs_c0 > mc_dmns_c0).sum())
        print('chrms > dmns:', (chrm_means.to_numpy() > mc_dmns_c0).sum())
        
        PlotUtil().show_grid()
        x = np.arange(len(self._chrids))
        markers = ['o', 's', 'p', 'P', '*']
        labels = ['chromosome', 'promoter bndrs', 
            'non-promoter bndrs', 'boundaries', 'domains']
        
        for i, y in enumerate((chrm_means, mc_prmtr_bndrs_c0, 
                mc_non_prmtr_bndrs_c0, mc_bndrs_c0, mc_dmns_c0)):
            plt.scatter(x, y, marker=markers[i], label=labels[i])

        plt.xticks(x, self._chrids)
        plt.xlabel('Chromosome')
        plt.ylabel('Mean C0')
        plt.title(f'Comparison of mean C0 in boundaries vs. domains')
        plt.legend()

        return IOUtil().save_figure(f'{PathUtil.get_figure_dir()}/mcdomains/bndrs_dmns_c0_{self}.png')
    
    def plot_bar_perc_in_prmtrs(self) -> Path:
        chrms = self._chrms
        mc_bndrs = chrms.apply(lambda chrm: HicExplBoundaries(chrm, self._res))
        perc_in_prmtrs = mc_bndrs.apply(
            lambda bndrs: Genes(bndrs._chrm)
                .in_promoter(bndrs.bndrs_df['middle']).mean() * 100)
        
        PlotUtil().show_grid()
        x = np.arange(len(self._chrids))
        plt.bar(x, perc_in_prmtrs)
        plt.xticks(x, self._chrids)
        plt.xlabel('Chromosome')
        plt.ylabel('Boundaries in promoters (%)')
        plt.title(f'Percentage of boundaries in promoters in chromosomes')
        plt.legend()
        return IOUtil().save_figure(f'{PathUtil.get_figure_dir()}/mcdomains/perc_bndrs_in_promoters_{self}.png')
    
    def num_bndrs_dmns(self) -> tuple[float, float]:  
        mc_bndrs = self._mc_bndrs
        num_bndrs = mc_bndrs.apply(lambda bndrs: len(bndrs.bndrs_df)).sum()
        num_dmns = num_bndrs - len(self._chrids)
        return num_bndrs, num_dmns
    
    def mean_dmn_len(self) -> float: 
        mc_bndrs = self._mc_bndrs
        return mc_bndrs.apply(
            lambda bndrs: bndrs.get_domains()['len'].sum()).sum()\
                / self.num_bndrs_dmns()[1]
    
    def individual_bndry_stat(self) -> None: 
        # TODO: Reduce function calls. Access index with .name if needed.  
        mc_bndrs = self._mc_bndrs
        num_mc_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: HicExplBoundaries.num_bndry_mean_c0_greater_than_dmn(bndrs)).sum()
        print('num_mc_bndrs_gt', num_mc_bndrs_gt)
        num_mc_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: HicExplBoundaries.num_prmtr_bndry_mean_c0_greater_than_dmn(bndrs)).sum()
        print('num_mc_prmtr_bndrs_gt', num_mc_prmtr_bndrs_gt)

        num_mc_non_prmtr_bndrs_gt = mc_bndrs.apply(
            lambda bndrs: HicExplBoundaries.num_non_prmtr_bndry_mean_c0_greater_than_dmns(bndrs)).sum()
        print('num_mc_non_prmtr_bndrs_gt', num_mc_non_prmtr_bndrs_gt)

        num_mc_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query('in_promoter'))
        ).sum()
        print('num_mc_prmtr_bndrs', num_mc_prmtr_bndrs)

        num_mc_non_prmtr_bndrs = mc_bndrs.apply(
            lambda bndrs: len(bndrs.bndrs_df.query('not in_promoter'))
        ).sum()
        print('num_mc_non_prmtr_bndrs', num_mc_non_prmtr_bndrs)
        

class MultiChrmHicExplBoundariesAggregator:
    def __init__(self, coll: MultiChrmHicExplBoundariesCollector):
        self._coll = coll
        self._agg_df = pd.DataFrame({'ChrIDs': [coll._coll_df['ChrID'].tolist()]})

    def _bndrs_gt_dmns(self):
        self._coll.save_stat([0,4])
        self._agg_df['b_gt_d'] = self._coll._coll_df[self._coll.col_for(4)].sum() \
            / self._coll._coll_df[self._coll.col_for(0)].sum() * 100
    
    def _p_bndrs_gt_dmns(self):
        self._coll.save_stat([7,5])
        self._agg_df['p_b_gt_d'] = self._coll._coll_df[self._coll.col_for(5)].sum() \
            / self._coll._coll_df[self._coll.col_for(7)].sum() * 100
    
    def _np_bndrs_gt_dmns(self):
        self._coll.save_stat([8,6])
        self._agg_df['np_b_gt_d'] = self._coll._coll_df[self._coll.col_for(6)].sum() \
            / self._coll._coll_df[self._coll.col_for(8)].sum() * 100
    
    def _num_bndrs(self):
        self._coll.save_stat([0, 7, 8])
        self._agg_df['num_b'] = self._coll._coll_df[self._coll.col_for(0)].sum()
        self._agg_df['num_p_b'] = self._coll._coll_df[self._coll.col_for(7)].sum()
        self._agg_df['num_np_b'] = self._coll._coll_df[self._coll.col_for(8)].sum()

    def save_stat(self):
        self._num_bndrs()
        self._bndrs_gt_dmns()
        self._p_bndrs_gt_dmns()
        self._np_bndrs_gt_dmns()
        self._agg_df['res'] = np.full((len(self._agg_df), ), self._coll._res)
        self._agg_df['lim'] = np.full((len(self._agg_df), ), self._coll._lim)
        self._agg_df['model'] = np.full((len(self._agg_df), ), str(self._coll._prediction))

        return IOUtil().append_tsv(self._agg_df, f'{PathUtil.get_data_dir()}/generated_data/mcdomains/aggr_mcdmns_stat.tsv')
