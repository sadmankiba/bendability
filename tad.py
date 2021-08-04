from util import IOUtil

import fanc
import fanc.plotting as fancplot
import matplotlib.pyplot as plt

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

'''
This module requires hic-matrix data that can be found here: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151553 
'''
hic_file = fanc.load("hic/data/GSE151553_A364_merged.juicer.hic@500")

class Tad:
    def get_boundaries(self):
        insulation_output_path = "data/generated_data/hic/yeast.insulation"
        if Path(insulation_output_path).is_file():
            insulation = fanc.load(insulation_output_path)
        else:
            IOUtil().make_parent_dirs(insulation_output_path)
        
            insulation = fanc.InsulationScores.from_hic(hic_file,
                            [1000, 2000, 5000, 10000, 25000],
                            file_name=insulation_output_path)
        boundaries = fanc.Boundaries.from_insulation_score(insulation, window_size=1000)
        ph = fancplot.TriangularMatrixPlot(hic_file, max_dist=50000, vmin=0, vmax=50)
        pb = fancplot.BarPlot(boundaries)
        f = fancplot.GenomicFigure([ph, pb])
        fig, axes = f.plot('XII:100kb-300kb')
        IOUtil().save_figure('figures/hic/xii_100kb_300kb.png')
        