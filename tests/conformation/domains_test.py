from matplotlib.pyplot import plot
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from conformation.domains import (
    BoundariesDomainsHEQuery,
    BoundariesHE,
    PlotBoundariesHE,
    IN_PROMOTER,
    LEFT,
    MEAN_C0,
    MIDDLE,
    MCBoundariesHEAggregator,
    MCBoundariesHECollector,
    RIGHT,
    SCORE,
)
from chromosome.chromosome import Chromosome
from models.prediction import Prediction
from chromosome.genes import Genes, Promoters


@pytest.fixture
def bndrs_vl(chrm_vl_mean7):
    return BoundariesHE(chrm_vl_mean7, res=500, lim=250)


class TestBoundariesHE:
    def test_read_boundaries_of(self, bndrs_vl: BoundariesHE):
        assert set(bndrs_vl.bndrs_df.columns) == set(
            [
                LEFT,
                RIGHT,
                SCORE,
                MIDDLE,
                IN_PROMOTER,
                MEAN_C0,
            ]
        )
        assert len(bndrs_vl.bndrs_df) == 57

    def test_mean_c0(self, bndrs_vl: BoundariesHE):
        mc0 = bndrs_vl.mean_c0
        assert mc0 == pytest.approx(-0.179, rel=1e-3)

    def test_add_mean_c0_col(self, bndrs_vl: BoundariesHE):
        mn = bndrs_vl.bndrs_df[MEAN_C0]
        assert np.all((mn > -0.7) & (mn < 0.2))

    def test_add_in_promoter_col(self, bndrs_vl: BoundariesHE):
        assert len(bndrs_vl.bndrs_df.query(IN_PROMOTER)) > len(
            bndrs_vl.bndrs_df.query(f"not {IN_PROMOTER}")
        )

    def test_prmtr_non_prmtr_bndrs(self, bndrs_vl: BoundariesHE):
        prmtr_bndrs = bndrs_vl.prmtr_bndrs()
        non_prmtr_bndrs = bndrs_vl.non_prmtr_bndrs()
        assert -0.5 < prmtr_bndrs.mean_c0 < 0
        assert -0.5 < non_prmtr_bndrs.mean_c0 < 0
        assert prmtr_bndrs.mean_c0 > non_prmtr_bndrs.mean_c0

        assert bndrs_vl.mean_c0 == pytest.approx(
            (
                prmtr_bndrs.mean_c0 * len(prmtr_bndrs)
                + non_prmtr_bndrs.mean_c0 * len(non_prmtr_bndrs)
            )
            / len(bndrs_vl),
            rel=1e-3,
        )


@pytest.mark.skip(reason="Updating domains")
class TestBoundariesDomainsHEQuery:
    def test_num_greater_than_dmns(self):
        bndrs = BoundariesHE(Chromosome("VI", Prediction(30)))
        bndrs_gt = bndrs.num_bndry_mean_c0_greater_than_dmn()
        prmtr_bndrs_gt = bndrs.num_prmtr_bndry_mean_c0_greater_than_dmn()
        non_prmtr_bndrs_gt = bndrs.num_non_prmtr_bndry_mean_c0_greater_than_dmns()
        assert bndrs_gt == prmtr_bndrs_gt + non_prmtr_bndrs_gt


@pytest.fixture
def plotbndrs_vl(chrm_vl):
    return PlotBoundariesHE(chrm_vl)


class TestPlotBoundariesHE:
    def test_prob_distrib_c0(self, plotbndrs_vl: PlotBoundariesHE):
        assert plotbndrs_vl.prob_distrib_c0().is_file()

    def test_line_c0_around(self, plotbndrs_vl: PlotBoundariesHE):
        assert plotbndrs_vl.line_c0_around().is_file()

    def test_plot_scatter_mean_c0_each_bndry(self, plotbndrs_vl: PlotBoundariesHE):
        figpath = plotbndrs_vl.scatter_mean_c0_at_indiv()
        assert figpath.is_file()

    def test_line_c0_around_indiv(self, plotbndrs_vl: PlotBoundariesHE):
        assert plotbndrs_vl._line_c0_around_indiv(plotbndrs_vl._bndrs[7], "").is_file()


@pytest.mark.skip(reason="Updating domains")
class TestMCBoundariesHECollector:
    def test_add_dmns_mean(self):
        coll = MCBoundariesHECollector(Prediction(30), ("VI", "VII"))
        c0_dmns_col = coll._add_dmns_mean()
        assert c0_dmns_col in coll._coll_df.columns
        assert all(coll._coll_df[c0_dmns_col] > -0.4)
        assert all(coll._coll_df[c0_dmns_col] < -0.1)

    def test_add_num_bndrs_gt_dmns(self):
        coll = MCBoundariesHECollector(Prediction(30), ("VI", "VII"))
        num_bndrs_gt_dmns_col = coll._add_num_bndrs_gt_dmns()
        assert num_bndrs_gt_dmns_col in coll._coll_df.columns
        assert all(coll._coll_df[num_bndrs_gt_dmns_col] > 10)
        assert all(coll._coll_df[num_bndrs_gt_dmns_col] < 200)

    def test_save_stat(self):
        # TODO *: Use default prediction 30
        coll = MCBoundariesHECollector(Prediction(30), ("VII", "X"))
        path = coll.save_stat([0, 3, 4])
        assert path.is_file()

    def test_plot_scatter_mean_c0(self):
        mcbndrs = MCBoundariesHECollector(Prediction(), ("VII", "XII", "XIII"))
        path = mcbndrs.plot_scatter_mean_c0()
        assert path.is_file()

    def test_plot_bar_perc_in_prmtrs(self):
        mcbndrs = MCBoundariesHECollector(Prediction(30), ("VII", "XII", "XIII"))
        path = mcbndrs.plot_bar_perc_in_prmtrs()
        assert path.is_file()

    def test_mean_dmn_len(self):
        """
        Test
            * If two mean boundaries are withing +-10%.
            * mean_dmn is within 7000 - 12000bp
        """
        mcbndrs_xi_ii = MCBoundariesHECollector(Prediction(30), ("XI", "II"))
        mcbndrs_vii = MCBoundariesHECollector(Prediction(30), ("VII",))
        xi_ii_mean_dmn = mcbndrs_xi_ii.mean_dmn_len()
        vii_mean_dmn = mcbndrs_vii.mean_dmn_len()
        assert xi_ii_mean_dmn > vii_mean_dmn * 0.9
        assert xi_ii_mean_dmn < vii_mean_dmn * 1.1
        assert xi_ii_mean_dmn > 7000
        assert xi_ii_mean_dmn < 12000


@pytest.mark.skip(reason="Updating domains")
class TestMCBoundariesHEAggregator:
    def test_save_stat(self):
        aggr = MCBoundariesHEAggregator(
            MCBoundariesHECollector(Prediction(30), ("IX", "XIV"))
        )
        path = aggr.save_stat()
        assert path.is_file()
