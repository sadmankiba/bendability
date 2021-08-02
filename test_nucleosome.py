from nucleosome import Nucleosome
from chromosome import Chromosome

from pathlib import Path

class TestNucleosome:
    def test_get_nuc_occupancy(self):
        nuc = Nucleosome(Chromosome('VII'))
        nuc_occ = nuc.get_nucleosome_occupancy()
        assert nuc_occ.shape == (nuc._chr._total_bp, )
        assert any(nuc_occ)

    def test_plot_c0_vs_dist_from_dyad_spread(self):
        nuc = Nucleosome(Chromosome('VL'))
        nuc.plot_c0_vs_dist_from_dyad_spread(150)
        path = Path('figures/nucleosome/dist_150_balanced_VL.png')
        assert path.is_file()