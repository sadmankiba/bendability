import pandas as pd
import matplotlib.pyplot as plt

from chromosome.chromosome import Chromosome
from chromosome.regions import PlotRegions, Regions, START, END
from util.constants import FigSubDir, GDataSubDir
from util.util import FileSave

class TestRegionsContain:
    def test_contains(self, chrm_vl_mean7: Chromosome):
        containers = pd.DataFrame({START: [3, 7, 9], END: [4, 12, 10]})
        rgns = Regions(chrm_vl_mean7, regions=containers)
        assert rgns._contains_loc([4, 11, 21, 3]).tolist() == [True, True, False]
        
    def test_save_regions(self, chrm_vl_mean7: Chromosome):
        rgns = Regions(chrm_vl_mean7, pd.DataFrame({START: [3], END: [10]}))    
        rgns.gdata_savedir = GDataSubDir.TEST
        assert rgns.save_regions().is_file()

class TestPlotRegions:
    def test_line_c0_indiv(self, chrm_vl_mean7: Chromosome):
        rgn = pd.Series({START: 5, END: 11})
        PlotRegions(chrm_vl_mean7).line_c0_indiv(rgn)
        FileSave.figure_in_figdir(f"{FigSubDir.TEST}/region.png")