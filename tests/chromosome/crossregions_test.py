from chromosome.chromosome import Chromosome
from chromosome.crossregions import CrossRegionsPlot

class TestCrossRegionsPlot:
    def test_prob_distrib_len(self, chrm_vl_mean7: Chromosome):
        assert CrossRegionsPlot(chrm_vl_mean7).prob_distrib_linkers_len_in_prmtrs().is_file()