import pytest 

from chromosome.chromosome import Chromosome
from chromosome.genes import Genes, Promoters, PromotersPlot


class TestGenes:
    def test_plot_mean_c0_vs_dist_from_dyad(self, chrm_vl: Chromosome):
        path = Genes(chrm_vl).plot_mean_c0_vs_dist_from_dyad()
        assert path.is_file()


@pytest.fixture
def prmtrs_vl(chrm_vl):
    return Promoters(chrm_vl)


class TestPromoters:
    def test_mean_c0(self, prmtrs_vl: Promoters):
        assert -0.3 < prmtrs_vl.mean_c0 < -0.1
        

@pytest.fixture
def prmtrsplt_vl(chrm_vl):
    return PromotersPlot(chrm_vl)

class TestPromotersPlot:
    def test_density(self, prmtrsplt_vl: PromotersPlot):
        assert prmtrsplt_vl.density_c0().is_file()

    def test_hist(self, prmtrsplt_vl: PromotersPlot):
        assert prmtrsplt_vl.hist_c0().is_file()