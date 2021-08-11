from prediction import Prediction
from chromosome import Chromosome
from tad import HicExplBoundaries, MultiChrmHicExplBoundaries
from genes import Genes

from numpy.testing import assert_almost_equal
import numpy as np


class TestHicExplBoundaries:
    def test_read_boundaries_of(self):
        bndrs = HicExplBoundaries(Chromosome('VIII'), res=500)
        assert set(bndrs.bndrs_df.columns) == set(['chromosome', 'left', 'right', 'id', 'score', 'middle'])
        assert len(bndrs.bndrs_df) == 53
    
    def test_get_domains(self):
        bndrs = HicExplBoundaries(Chromosome('XV'))
        dmns_df = bndrs.get_domains()
        assert len(dmns_df) == len(bndrs.bndrs_df) - 1
        assert dmns_df.iloc[9]['start'] == bndrs.bndrs_df.iloc[9]['right']
        assert dmns_df.iloc[9]['end'] == bndrs.bndrs_df.iloc[10]['left']

    def test_add_mean_c0_col(self):
        bndrs = HicExplBoundaries(Chromosome('X', Prediction(30)))
        mn = bndrs.add_mean_c0_col()['mean_c0']
        assert np.all((mn > -0.7) & (mn < 0.2))

    def test_bndry_domain_mean_c0(self):
        bndrs = HicExplBoundaries(Chromosome('VL'))
        bndry_c0, dmn_c0 = bndrs.bndry_domain_mean_c0()
        assert -0.3 < bndry_c0 < 0
        assert -0.3 < dmn_c0 < 0
    
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
        num_prmtr_bndrs = Genes(chrm).in_promoter(bndrs.bndrs_df['middle']).sum()
        num_non_prmtr_bndrs = len(bndrs.bndrs_df) - num_prmtr_bndrs
        assert_almost_equal(bndrs_c0, (prmtr_bndry_c0 * num_prmtr_bndrs 
            + non_prmtr_bndry_c0 * num_non_prmtr_bndrs) / len(bndrs.bndrs_df),
            decimal=6)


class TestMultiChrmHicExplBoundaries:
    def test_plot_scatter_mean_c0(self):
        mcbndrs = MultiChrmHicExplBoundaries(Prediction(), ('VII','XII','XIII'))
        path = mcbndrs.plot_scatter_mean_c0()
        assert path.is_file()
    
    def test_plot_bar_perc_in_prmtrs(self):
        mcbndrs = MultiChrmHicExplBoundaries(Prediction(30), ('VII','XII','XIII'))
        path = mcbndrs.plot_bar_perc_in_prmtrs()
        assert path.is_file()
    
    def test_mean_dmn_len(self):
        """
        Test 
            * If two mean boundaries are withing +-10%. 
            * mean_dmn is within 7000 - 12000bp
        """
        mcbndrs_xi_ii = MultiChrmHicExplBoundaries(Prediction(30), ('XI', 'II'))
        mcbndrs_vii = MultiChrmHicExplBoundaries(Prediction(30), ('VII'))
        xi_ii_mean_dmn = mcbndrs_xi_ii.mean_dmn_len()
        vii_mean_dmn = mcbndrs_vii.mean_dmn_len()
        assert xi_ii_mean_dmn > vii_mean_dmn * 0.9
        assert xi_ii_mean_dmn < vii_mean_dmn * 1.1
        assert xi_ii_mean_dmn > 7000
        assert xi_ii_mean_dmn < 12000
    