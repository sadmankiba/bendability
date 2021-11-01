import pytest

from models.prediction import Prediction
from chromosome.nucleosomes import Nucleosomes
from chromosome.chromosome import Chromosome
from util.constants import CHRV_TOTAL_BP

@pytest.fixture
def nucs_vl(chrm_vl_mean7):
    return Nucleosomes(Chromosome('VL'))


class TestNucleosomes:
    def test_get_nuc_occupancy(self, nucs_vl: Nucleosomes):
        nuc_occ = nucs_vl.get_nucleosome_occupancy()
        assert nuc_occ.shape == (nucs_vl._chrm.total_bp,)
        assert any(nuc_occ)

    def test_plot_c0_vs_dist_from_dyad_spread(self, nucs_vl: Nucleosomes):
        path = nucs_vl.plot_c0_vs_dist_from_dyad_spread(150)
        assert path.is_file()

    def test_dyads_between(self, nucs_vl: Nucleosomes):
        dyad_arr = nucs_vl.dyads_between(50000, 100000)
        assert 200 < dyad_arr.size < 300

        rev_dyad_arr = nucs_vl.dyads_between(50000, 100000, -1)
        assert list(rev_dyad_arr) == list(dyad_arr[::-1])

    def test_get_nuc_regions(self, nucs_vl: Nucleosomes):
        nucs_cvr = nucs_vl.get_nuc_regions()
        assert nucs_cvr.shape == (CHRV_TOTAL_BP,)
