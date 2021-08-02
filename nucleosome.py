from chromosome import Chromosome
from reader import DNASequenceReader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path 
import time

from util import IOUtil

class Nucleosome:
    """
    Abstraction of nucleosomes in a chromosome
    """
    def __init__(self, chr: Chromosome):
        self._chr = chr
        self._nuc_df = DNASequenceReader().read_nuc_center()
    

    def _plot_c0_vs_dist_from_dyad(self, x: np.ndarray, y: np.ndarray, dist: int, spread_str: str) -> None:
        """Underlying plotter of c0 vs dist from dyad"""
        # Plot C0
        plt.plot(x, y, color='tab:blue')
        
        # Highlight nuc. end positions and dyad
        y_lim = plt.gca().get_ylim()
        for p in [-73, 73]:
            plt.axvline(x=p, color='tab:green', linestyle='--')
            plt.text(p, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, f'{p}bp', color='tab:green', ha='left', va='center')

        plt.axvline(x=0, color='tab:orange', linestyle='--')
        plt.text(0, y_lim[0] + (y_lim[1] - y_lim[0]) * 0.75, f'dyad', color='tab:orange', ha='left', va='center')

        self._chr.plot_avg()
        plt.grid()

        plt.xlabel('Distance from dyad(bp)')
        plt.ylabel('C0')
        plt.title(f'C0 of +-{dist} bp from nuclesome dyad')
        
        # Save figure
        IOUtil().save_figure(f'figures/nucleosome/dist_{dist}_{spread_str}_{self._chr._chr_id}.png')
        

    def plot_c0_vs_dist_from_dyad_no_spread(self, dist=150) -> None:
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome V from 50-bp sequence C0

        Currently, giving horizontally shifted graph. (incorrect)
         
        Args: 
            dist: +-distance from dyad to plot (1-indexed)
        """
        centers = self._nuc_df.loc[self._nuc_df['Chromosome ID'] == f'chr{self._chr._chr_num}']['Position'].tolist()
        
        # Read C0 of -dist to +dist sequences
        c0_at_nuc: list[list[float]] = list(
            map(
                lambda c: self._chr.read_chr_lib_segment(c - dist, c + dist)['C0'].tolist(), 
                centers
            )
        )

        # Make lists of same length
        min_len = min(map(len, c0_at_nuc))
        max_len = max(map(len, c0_at_nuc))
        assert max_len - min_len <= 1
        c0_at_nuc = list(
            map(
                lambda l: l[:min_len],
                c0_at_nuc
            )
        ) 
        mean_c0 = np.array(c0_at_nuc).mean(axis=0)
        x = (np.arange(mean_c0.size) - mean_c0.size / 2) * 7 + 1

        self._plot_c0_vs_dist_from_dyad(x, mean_c0, dist, 'no_spread')
    

    # *** #
    def plot_c0_vs_dist_from_dyad_spread(self, dist=150) -> None:
        """
        Plot C0 vs. distance from dyad of nucleosomes in chromosome by
        spreading 50-bp sequence C0

        Args: 
            dist: +-distance from dyad to plot (1-indexed) 
        """
        spread_c0 = self._chr.spread_c0_balanced()
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[nuc_df['Chromosome ID'] == f'chr{self._chr._chr_num}']['Position'].tolist()
        
        # Remove center positions at each end that aren't in at least dist depth
        centers = list(filter(lambda i: i > dist and i < self._chr._total_bp - dist, centers))

        # Read C0 of -dist to +dist sequences
        c0_at_nuc: list[np.ndarray] = list(
            map(
                lambda c: spread_c0[c - 1 - dist:c + dist], 
                centers
            )
        )
        assert c0_at_nuc[0].size == 2 * dist + 1
        assert c0_at_nuc[-1].size == 2 * dist + 1
        
        x = np.arange(dist * 2 + 1) - dist
        mean_c0 = np.array(c0_at_nuc).mean(axis=0)

        self._plot_c0_vs_dist_from_dyad(x, mean_c0, dist, 'balanced')


    def get_nucleosome_occupancy(self) -> np.ndarray:
        """Returns estimated nucleosome occupancy across whole chromosome

        Each dyad is extended 50 bp in both direction, resulting in a footprint
        of 101 bp for each nucleosome.
        """
        saved_data = Path(f'data/generated_data/nucleosome/nuc_occ_{self._chr._chr_id}.tsv')
        if saved_data.is_file():
            return pd.read_csv(saved_data, sep='\t')['nuc_occupancy'].to_numpy()
        
        nuc_df = DNASequenceReader().read_nuc_center()
        centers = nuc_df.loc[nuc_df['Chromosome ID'] == f'chr{self._chr._chr_num}']['Position'].tolist()
        
        t = time.time()
        nuc_occ = np.full((self._chr._total_bp, ), fill_value=0)
        for c in centers:
            nuc_occ[c - 1 - 50: c + 50] = 1
        
        print('Calculation of spread c0 balanced:', time.time() - t, 'seconds.')
        
        # Save data
        if not saved_data.parents[0].is_dir():
            saved_data.parents[0].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'position': np.arange(self._chr._total_bp) + 1, 'nuc_occupancy': nuc_occ})\
            .to_csv(saved_data, sep='\t', index=False)

        return nuc_occ
