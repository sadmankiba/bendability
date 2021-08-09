from chromosome import Chromosome
from genes import Genes

class TestGenes:
    def test_plot_mean_c0_vs_dist_from_dyad(self):
        path = Genes(Chromosome('VL')).plot_c0_vs_dist_from_dyad() 
        assert path.is_file()