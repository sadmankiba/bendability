from __future__ import annotations
from genes import Genes

from util import IOUtil, PlotUtil
from custom_types import ChrId, PositiveInt, YeastChrNum
from chromosome import Chromosome
from prediction import Prediction
from constants import ChrIdList

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

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

    def _read_boundaries(self) -> pd.DataFrame:
        return pd.read_table(f'data/input_data/domains/'
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

    def add_mean_c0_col(self) -> None:
        if 'mean_c0' not in self.bndrs_df.columns.tolist():
            self.bndrs_df = self.bndrs_df.assign(mean_c0=lambda df:
                self._chrm.mean_c0_at_bps(df['middle'], self._lim, self._lim))

    def add_in_promoter_col(self) -> None:
        if 'in_promoter' not in self.bndrs_df.columns.tolist():
            self.bndrs_df = self.bndrs_df.assign(in_promoter=lambda df: 
                Genes(self._chrm).in_promoter(df['middle']))

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
        self.add_in_promoter_col()
        return self._chrm.mean_c0_of_segments(
            self._bndrs_df.iloc[self._bndrs_df['in_promoter'].to_numpy()]
                ['middle'], self._lim, self._lim)
        
    def non_prmtr_bndrs_mean_c0(self) -> float: 
        self.add_in_promoter_col()
        return self._chrm.mean_c0_of_segments(
            self.bndrs_df.iloc[~(self.bndrs_df['in_promoter'].to_numpy())]['middle'], self._lim, self._lim)

    def num_bndry_mean_c0_greater_than_dmn(self) -> PositiveInt:
        self.add_mean_c0_col()
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df['mean_c0'] > dmns_mean_c0).sum()

    def num_prmtr_bndry_mean_c0_greater_than_dmn(self) -> float:
        self.add_mean_c0_col()
        self.add_in_promoter_col()
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df.query('in_promoter')['mean_c0'] > dmns_mean_c0).sum()
    
    def num_non_prmtr_bndry_mean_c0_greater_than_dmns(self) -> float:
        self.add_mean_c0_col()
        self.add_in_promoter_col()
        _, dmns_mean_c0 = self.bndry_domain_mean_c0()
        return (self.bndrs_df.query('not in_promoter')['mean_c0'] > dmns_mean_c0).sum()



# TODO: Create common MultiChrm Class for loops and boundaries
class MultiChrmHicExplBoundaries:
    def __init__(self, 
                prediction: Prediction, 
                chrids: tuple[ChrId] = ChrIdList, 
                res: PositiveInt = 400,
                lim: PositiveInt = 200):
        self._prediction = prediction
        self._chrids = chrids
        self._res = res
        self._lim = lim

    def __str__(self):
        ext = 'with_vl' if 'VL' in self._chrids else 'without_vl'
        return f'res_{self._res}_lim_{self._lim}_md_{self._prediction}_{ext}'

    def _get_chrms(self) -> pd.Series:
        return pd.Series(list(map(
            lambda chrm_id: Chromosome(chrm_id, self._prediction), self._chrids)))

    def _get_mc_bndrs(self) -> pd.Series:
        return self._get_chrms().apply(lambda chrm: HicExplBoundaries(chrm, self._res, self._lim))

    def plot_scatter_mean_c0(self) -> Path:
        """Draw scatter plot of mean c0 at boundaries and domains of
        chromosomes"""
        chrms = self._get_chrms()
        chrm_means = chrms.apply(lambda chrm: chrm.get_spread().mean())
        
        mc_bndrs = self._get_mc_bndrs()
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

        return IOUtil().save_figure(f'figures/mcdomains/bndrs_dmns_c0_{self}.png')
    
    def plot_bar_perc_in_prmtrs(self) -> Path:
        chrms = self._get_chrms()
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
        return IOUtil().save_figure(f'figures/mcdomains/perc_bndrs_in_promoters_{self}.png')
    
    def num_bndrs_dmns(self) -> tuple[float, float]:  
        mc_bndrs = self._get_mc_bndrs()
        num_bndrs = mc_bndrs.apply(lambda bndrs: len(bndrs.bndrs_df)).sum()
        num_dmns = num_bndrs - len(self._chrids)
        return num_bndrs, num_dmns
    
    def mean_dmn_len(self) -> float: 
        mc_bndrs = self._get_mc_bndrs()
        return mc_bndrs.apply(
            lambda bndrs: bndrs.get_domains()['len'].sum()).sum()\
                / self.num_bndrs_dmns()[1]
    
    def individual_bndry_stat(self) -> None: 
        mc_bndrs = self._get_mc_bndrs()
        bndrs_df = mc_bndrs
        mc_bndrs.apply()
        