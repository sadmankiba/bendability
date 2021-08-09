from prediction import Prediction
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
    
    def test_dyads_between(self):
        nucs = Nucleosome(Chromosome('IV'))
        dyad_arr = nucs.dyads_between(50000, 100000)
        assert 200 < dyad_arr.size < 300
        
        rev_dyad_arr = nucs.dyads_between(50000, 100000, -1)
        assert list(rev_dyad_arr) == list(dyad_arr[::-1])
