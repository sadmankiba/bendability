import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from conformation.tad import (
    BoundariesHE,
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
from chromosome.genes import Genes


@pytest.mark.skip(reason="Updating tad")


class TestBoundariesHE:
    def test_read_boundaries_of(self):
        bndrs = BoundariesHE(Chromosome("VIII", Prediction(30)), res=500)
        assert set(bndrs.bndrs_df.columns) == set(
            [
                LEFT,
                RIGHT,
                SCORE,
                MIDDLE,
                IN_PROMOTER,
                MEAN_C0,
            ]
        )
        assert len(bndrs.bndrs_df) == 53

    def test_get_domains(self):
        bndrs = BoundariesHE(Chromosome("XV", Prediction(30)))
        dmns_df = bndrs.get_domains()
        assert len(dmns_df) == len(bndrs.bndrs_df) - 1
        assert dmns_df.iloc[9]["start"] == bndrs.bndrs_df.iloc[9]["right"]
        assert dmns_df.iloc[9]["end"] == bndrs.bndrs_df.iloc[10]["left"]

    def test_add_mean_c0_col(self):
        bndrs = BoundariesHE(Chromosome("X", Prediction(30)))
        mn = bndrs.bndrs_df["mean_c0"]
        assert np.all((mn > -0.7) & (mn < 0.2))

    def test_add_in_promoter_col(self):
        bndrs = BoundariesHE(Chromosome("X", Prediction(30)))
        assert len(bndrs.bndrs_df.query("in_promoter")) > len(
            bndrs.bndrs_df.query("not in_promoter")
        )

    def test_bndry_domain_mean_c0(self):
        bndrs = BoundariesHE(Chromosome("VL"))
        bndry_c0, dmn_c0 = bndrs.bndry_domain_mean_c0()
        assert -0.3 < bndry_c0 < 0
        assert -0.3 < dmn_c0 < 0

    def test_prmtr_non_prmtr_mean(self):
        chrm = Chromosome("VI", Prediction(30))
        bndrs = BoundariesHE(chrm)
        prmtr_bndry_c0 = bndrs.prmtr_bndrs_mean_c0()
        non_prmtr_bndry_c0 = bndrs.non_prmtr_bndrs_mean_c0()
        assert -0.5 < prmtr_bndry_c0 < 0
        assert -0.5 < non_prmtr_bndry_c0 < 0
        assert prmtr_bndry_c0 > non_prmtr_bndry_c0

        # Check mathematically if two groups make up total boundaries mean
        bndrs_c0, _ = bndrs.bndry_domain_mean_c0()
        num_prmtr_bndrs = Genes(chrm).in_promoter(bndrs.bndrs_df["middle"]).sum()
        num_non_prmtr_bndrs = len(bndrs.bndrs_df) - num_prmtr_bndrs
        assert_almost_equal(
            bndrs_c0,
            (
                prmtr_bndry_c0 * num_prmtr_bndrs
                + non_prmtr_bndry_c0 * num_non_prmtr_bndrs
            )
            / len(bndrs.bndrs_df),
            decimal=6,
        )

    def test_num_greater_than_dmns(self):
        bndrs = BoundariesHE(Chromosome("VI", Prediction(30)))
        bndrs_gt = bndrs.num_bndry_mean_c0_greater_than_dmn()
        prmtr_bndrs_gt = bndrs.num_prmtr_bndry_mean_c0_greater_than_dmn()
        non_prmtr_bndrs_gt = bndrs.num_non_prmtr_bndry_mean_c0_greater_than_dmns()
        assert bndrs_gt == prmtr_bndrs_gt + non_prmtr_bndrs_gt

    def test_plot_scatter_mean_c0_each_bndry(self):
        bndrs = BoundariesHE(Chromosome("VL"))
        path = bndrs.plot_scatter_mean_c0_each_bndry()
        assert path.is_file()


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


class TestMCBoundariesHEAggregator:
    def test_save_stat(self):
        aggr = MCBoundariesHEAggregator(
            MCBoundariesHECollector(Prediction(30), ("IX", "XIV"))
        )
        path = aggr.save_stat()
        assert path.is_file()
