import pytest

from chromosome.chromosome import Chromosome
from chromosome.genes import Genes, Promoters, PromotersPlot
from conformation.domains import BoundariesHE, MIDDLE


class TestGenes:
    def test_plot_mean_c0_vs_dist_from_dyad(self, chrm_vl: Chromosome):
        path = Genes(chrm_vl).plot_mean_c0_vs_dist_from_dyad()
        assert path.is_file()


@pytest.fixture
def prmtrs_vl(chrm_vl_mean7):
    return Promoters(chrm_vl_mean7, ustr_tss=500, dstr_tss=0)


class TestPromoters:
    def test_mean_c0(self, prmtrs_vl: Promoters):
        assert -0.3 < prmtrs_vl.mean_c0 < -0.1

    def test_and_x(self, prmtrs_vl: Promoters):
        bndrs = BoundariesHE(prmtrs_vl.chrm, 500, 250)
        prmtrs_wb = prmtrs_vl.and_x(bndrs[MIDDLE], True)
        prmtrs_wob = prmtrs_vl.and_x(bndrs[MIDDLE], False)
        assert len(prmtrs_wb) + len(prmtrs_wob) == len(prmtrs_vl)

        assert (
            pytest.approx(
                (
                    prmtrs_wb.mean_c0 * len(prmtrs_wb)
                    + prmtrs_wob.mean_c0 * len(prmtrs_wob)
                )
                / len(prmtrs_vl),
                rel=2e-3,
            )
            == prmtrs_vl.mean_c0
        )


@pytest.fixture
def prmtrsplt_vl(chrm_vl_mean7):
    return PromotersPlot(chrm_vl_mean7)


class TestPromotersPlot:
    def test_prob_distrib(self, prmtrsplt_vl: PromotersPlot):
        assert prmtrsplt_vl.prob_distrib_c0().is_file()

    def test_hist(self, prmtrsplt_vl: PromotersPlot):
        assert prmtrsplt_vl.hist_c0().is_file()
