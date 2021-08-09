from pathlib import Path

from util import IOUtil, PlotUtil
import matplotlib.pyplot as plt
import numpy as np

from chromosome import Chromosome
from reader import GeneReader
from nucleosome import Nucleosome

class Genes:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm 
        self._tr_df = GeneReader().read_transcription_regions_of(chrm._chr_num)
        self._tr_df = self._add_dyads_in_tr(self._tr_df)

    def _add_dyads_in_tr(self, tr_df):
        nucs = Nucleosome(self._chrm)
        tr_df['dyads'] = self._tr_df.apply(
            lambda tr: nucs.dyads_between(tr['start'], tr['end'], tr['strand']), 
            axis=1)
        
        return tr_df 
    
    def plot_c0_vs_dist_from_dyad(self) -> Path:
        p1_dyads = self._tr_df['dyads'].apply(lambda dyads: dyads[0])
        mean_c0 = self._chrm.mean_c0_around_bps(p1_dyads, 600, 400)
        
        plt.close()
        plt.clf()
        PlotUtil().show_grid()
        plt.plot(np.arange(-600, 400 + 1), mean_c0)

        plt.xlabel('Distance from dyad (bp)')
        plt.ylabel('Mean C0')
        plt.title(f'{self._chrm._c0_type} Mean C0 around +1 dyad'
                f' in chromosome {self._chrm._chr_num}')

        return IOUtil().save_figure(f'figures/gene/dist_p1_dyad_{self._chrm}.png')
