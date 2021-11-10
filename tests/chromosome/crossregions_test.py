import pytest

from chromosome.chromosome import Chromosome
from chromosome.crossregions import CrossRegionsPlot
from chromosome.genes import Promoters


@pytest.fixture
def crplt_vl(chrm_vl_mean7: Chromosome):
    return CrossRegionsPlot(chrm_vl_mean7)


class TestCrossRegionsPlot:
    def test_line_c0_prmtr_indiv_toppings(self, crplt_vl: CrossRegionsPlot, prmtrs_vl: Promoters):
        assert crplt_vl._line_c0_prmtr_indiv_toppings(prmtrs_vl[10]).is_file() 
        
    def test_line_c0_toppings(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.line_c0_toppings(342000, 343000).is_file()

    def test_text_pos_calc(self, crplt_vl: CrossRegionsPlot):
        assert list(crplt_vl._text_pos_calc(5, 24, 0.04)) == [
            (5, 0.04),
            (9, 0.02),
            (13, -0.02),
            (17, -0.04),
            (21, 0.04),
        ]

    def test_prob_distrib_bndrs_nearest_ndr_distnc(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.prob_distrib_bndrs_nearest_ndr_distnc().is_file()

    def test_distrib_cuml_bndrs_nearest_ndr_distnc(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.distrib_cuml_bndrs_nearest_ndr_distnc().is_file()

    def test_num_prmtrs_bndrs_ndrs(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.num_prmtrs_bndrs_ndrs().is_file()

    def test_prob_distrib_prmtrs_ndrs(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.prob_distrib_prmtr_ndrs().is_file()

    def test_prob_distrib_len(self, crplt_vl: CrossRegionsPlot):
        assert crplt_vl.prob_distrib_linkers_len_in_prmtrs().is_file()
