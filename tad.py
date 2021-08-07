from __future__ import annotations

from util import IOUtil
from custom_types import ChrId, YeastChrNum
from chromosome import ChrIdList, Chromosome
from prediction import Prediction

import fanc
import fanc.plotting as fancplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
'''
This module requires hic-matrix data that can be found here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151553 
'''


class Boundary:
    def __init__(self, resolution: int = 500, window_size: int = 1000):
        self._hic_file = fanc.load(
            f"hic/data/GSE151553_A364_merged.juicer.hic@{resolution}")
        self._window_size = window_size
        self._resolution = resolution
        self.boundaries = self._get_all_boundaries()

    def plot_boundaries(self, chrm_num: YeastChrNum, start: int, end: int):
        ph = fancplot.TriangularMatrixPlot(self._hic_file,
                                           max_dist=50000,
                                           vmin=0,
                                           vmax=50)
        pb = fancplot.BarPlot(self.boundaries)
        f = fancplot.GenomicFigure([ph, pb])
        if start > 0 and end > 0:
            fig, axes = f.plot(
                f'{chrm_num}:{int(start / 1000)}kb-{int(end / 1000)}kb')
            IOUtil().save_figure('figures/hic/xii_100kb_300kb.png')
        else:
            fig, axes = f.plot(f'{chrm_num}')
            IOUtil().save_figure(f'figures/hic/{chrm_num}_boundaries.png')

    def _get_insulation(self) -> fanc.InsulationScores:
        insulation_output_path = f"data/generated_data/hic/insulation_fanc_res_{self._resolution}"
        if Path(insulation_output_path).is_file():
            return fanc.load(insulation_output_path)

        IOUtil().make_parent_dirs(insulation_output_path)

        return fanc.InsulationScores.from_hic(self._hic_file,
                                              [1000, 2000, 5000, 10000, 25000],
                                              file_name=insulation_output_path)

    def _get_all_boundaries(self) -> fanc.architecture.domains.Boundaries:
        boundary_file_path = f'data/generated_data/hic/boundaries_fanc_res_{self._resolution}_w_{self._window_size}'
        if Path(boundary_file_path).is_file():
            return fanc.load(boundary_file_path)

        insulation = self._get_insulation()
        return fanc.Boundaries.from_insulation_score(
            insulation,
            window_size=self._window_size,
            file_name=boundary_file_path)

    def save_boundaries(self) -> None:
        df = pd.DataFrame({
            'chromosome':
            list(map(lambda r: r.chromosome, self.boundaries)),
            'start':
            list(map(lambda r: r.start, self.boundaries)),
            'end':
            list(map(lambda r: r.end, self.boundaries)),
            'score':
            list(map(lambda r: r.score, self.boundaries))
        })
        IOUtil().save_tsv(
            df,
            f'data/generated_data/hic/boundaries_w_{self._window_size}_res_{self._resolution}.tsv'
        )

    def get_boundaries_in(self,
                          chrm_num: YeastChrNum) -> list[fanc.GenomicRegion]:
        return list(
            filter(lambda region: chrm_num == region.chromosome,
                   self.boundaries))

    def _get_octiles(self, regions: list[fanc.GenomicRegion] = None):
        """
        Returns: 
            A numpy 1D array of size 9 denoting octiles of scores
        """
        if regions is None:
            regions = self.boundaries
        scores = np.array(list(map(lambda r: r.score, regions)))
        scores = scores[~np.isnan(scores)]
        return np.percentile(scores, np.arange(9) / 8 * 100)

    def get_one_eighth_regions_in(
            self, chrm_num: YeastChrNum) -> list[list[fanc.GenomicRegion]]:
        regions_chrm = self.get_boundaries_in(chrm_num)
        octiles = self._get_octiles(regions_chrm)
        return [
            list(
                filter(lambda r: octiles[i] < r.score <= octiles[i + 1],
                       regions_chrm)) for i in range(8)
        ]

    def save_global_octiles_of_scores(self) -> None:
        octiles = self._get_octiles()
        oct_cols = [f'octile_{num}' for num in range(9)]
        IOUtil().save_tsv(
            pd.DataFrame(octiles.reshape((1, 9)), columns=oct_cols),
            'data/generated_data/hic/boundary_octiles.tsv')


class BoundaryAnalysis:
    def __init__(self) -> None:
        self._boundary = Boundary()

    def _find_c0_at_octiles_of(self, chrm_id: ChrId, lim) -> None:
        """
        Find c0 at octiles of boundaries in each chromosome
        """
        # First, find for chr V actual
        chrm = Chromosome(chrm_id, Prediction(30))
        one_eighth_regions = self._boundary.get_one_eighth_regions_in(
            chrm._chr_id)
        assert len(one_eighth_regions) == 8

        c0_spread = chrm.get_spread()

        def _mean_at(position: int) -> float:
            """Calculate mean C0 of a region around a position"""
            return c0_spread[position - 1 - lim:position + lim].mean()

        one_eighth_centers = [
            list(map(lambda r: r.center, one_eighth_regions[i]))
            for i in range(8)
        ]
        one_eighth_means = [
            np.mean(
                list(
                    map(lambda center: _mean_at(int(center)),
                        one_eighth_centers[i]))) for i in range(8)
        ]
        cols = ['avg', 'lim'] + [f'eighth_{num}' for num in range(1, 9)]
        vals = np.concatenate(([c0_spread.mean(), lim], one_eighth_means))
        IOUtil().append_tsv(pd.DataFrame(vals.reshape((1, -1)), columns=cols),
                            'data/generated_data/hic/mean_at_boundaries.tsv')

    def find_c0_at_octiles(self, lim: int = 250):
        for chrm_id in ChrIdList:
            self._find_c0_at_octiles_of(chrm_id, lim)
