import numpy as np

from chromosome.chromosome import Chromosome
from chromosome.genes import Genes, Promoters


class TestGenes:
    def test_plot_mean_c0_vs_dist_from_dyad(self):
        path = Genes(Chromosome("VL")).plot_mean_c0_vs_dist_from_dyad()
        assert path.is_file()

    def test_in_promoter(self):
        genes = Genes(Chromosome("II"))
        assert list(
            genes.in_promoter([741800, 742000, 742500, 634700, 636900, 636400])
        ) == [False, False, True, True, False, False]


class TestPromoters:
    def test_mean_c0(self, chrm_vl: Chromosome):
        assert -0.5 < Promoters(chrm_vl).mean_c0 < 0.2