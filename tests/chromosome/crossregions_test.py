import pytest

from chromosome.chromosome import Chromosome
from chromosome.crossregions import CrossRegionsPlot

@pytest.fixture
def crplt_vl(chrm_vl_mean7: Chromosome):
    return CrossRegionsPlot(chrm_vl_mean7)

class TestCrossRegionsPlot:
    def test_num_prmtrs_bndrs_ndrs(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.num_prmtrs_bndrs_ndrs().is_file()

    def test_prob_distrib_prmtrs_ndrs(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.prob_distrib_prmtr_ndrs().is_file()

    def test_prob_distrib_len(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.prob_distrib_linkers_len_in_prmtrs().is_file()