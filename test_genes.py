import numpy as np 

from chromosome import Chromosome
from genes import Genes

class TestGenes:
    def test_plot_mean_c0_vs_dist_from_dyad(self):
        path = Genes(Chromosome('VL')).plot_mean_c0_vs_dist_from_dyad() 
        assert path.is_file()
    
    def test_frwrd_rvrs(self):
        genes = Genes(Chromosome('IX'))
        frwrd_tr_df = genes._frwrd_tr_df()
        assert np.all(frwrd_tr_df['strand'] == 1)
        
        rvrs_tr_df = genes._rvrs_tr_df()
        assert np.all(rvrs_tr_df['strand'] == -1)

        assert len(frwrd_tr_df) + len(rvrs_tr_df) == len(genes._tr_df)

    def test_in_promoter(self):
        genes = Genes(Chromosome('II'))
        assert list(genes.in_promoter([741800, 742000, 742500, 634700, 636900, 636400]))\
            == [False, False, True, True, False, False]