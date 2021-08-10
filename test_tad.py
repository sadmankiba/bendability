from prediction import Prediction
from chromosome import Chromosome
from tad import FancBoundary, HicExplBoundaries, MultiChrmHicExplBoundaries
from custom_types import YeastChrNum
from genes import Genes

from numpy.testing import assert_almost_equal


class TestFancBoundary:
    def test_get_boundaries(self):
        boundaries = FancBoundary()._get_all_boundaries()
        assert len(boundaries) > 0
        chrm, region = boundaries[0].split(":")
        assert chrm in YeastChrNum
        resolution_start, resolution_end = region.split("-")
        assert int(resolution_start) >= 0
        assert int(resolution_end) >= 0

    def test_get_boundaries_in(self):
        regions = FancBoundary().get_boundaries_in('XIII')
        assert len(regions) > 0
        a_region = regions[0]
        assert a_region.score > 0
        assert a_region.chromosome == 'XIII'
        assert type(a_region.start) == int
        assert a_region.start > 0
        assert a_region.end > 0
        assert a_region.center > 0


class TestHicExplBoundaries:
    def test_read_boundaries_of(self):
        bndrs = HicExplBoundaries(Chromosome('VIII'))
        assert set(bndrs._bndrs_df.columns) == set(['chromosome', 'left', 'right', 'id', 'score', 'middle'])
        assert len(bndrs._bndrs_df) == 65
    
    def test_bndry_domain_mean_c0(self):
        bndrs = HicExplBoundaries(Chromosome('VL'))
        bndry_c0, dmn_c0 = bndrs.bndry_domain_mean_c0()
        assert -0.3 < bndry_c0 < 0
        assert -0.3 < dmn_c0 < 0
        assert bndry_c0 > dmn_c0
    
    def test_prmtr_non_prmtr_mean(self):
        chrm = Chromosome('VI', Prediction(30))
        bndrs = HicExplBoundaries(chrm)
        prmtr_bndry_c0 = bndrs.prmtr_bndrs_mean_c0()
        non_prmtr_bndry_c0 = bndrs.non_prmtr_bndrs_mean_c0()
        assert -0.5 < prmtr_bndry_c0 < 0
        assert -0.5 < non_prmtr_bndry_c0 < 0
        assert prmtr_bndry_c0 > non_prmtr_bndry_c0

        # Check mathematically if two groups make up total boundaries mean
        bndrs_c0, _ = bndrs.bndry_domain_mean_c0()
        num_prmtr_bndrs = Genes(chrm).in_promoter(bndrs._bndrs_df['middle']).sum()
        num_non_prmtr_bndrs = len(bndrs._bndrs_df) - num_prmtr_bndrs
        assert_almost_equal(bndrs_c0, (prmtr_bndry_c0 * num_prmtr_bndrs 
            + non_prmtr_bndry_c0 * num_non_prmtr_bndrs) / len(bndrs._bndrs_df),
            decimal=6)


class TestMultiChrmHicExplBoundaries:
    def test_plot_scatter_mean_c0(self):
        mcbndrs = MultiChrmHicExplBoundaries(Prediction(), ('VII','XII','XIII'))
        path = mcbndrs.plot_scatter_mean_c0()
        assert path.is_file()
    
    def test_plot_bar_perc_in_prmtrs(self):
        mcbndrs = MultiChrmHicExplBoundaries(Prediction(), ('VII','XII','XIII'))
        path = mcbndrs.plot_bar_perc_in_prmtrs()
        assert path.is_file()
    