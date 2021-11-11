import pytest

from chromosome.chromosome import Chromosome
from conformation.domains import BoundariesHE, BndParm


@pytest.fixture
def chrm_vl():
    return Chromosome("VL")


@pytest.fixture
def chrm_vl_mean7():
    return Chromosome("VL", spread_str="mean7")


@pytest.fixture
def chrm_i():
    return Chromosome("I")

@pytest.fixture
def bndrs_hirs_vl(chrm_vl_mean7: Chromosome):
    return BoundariesHE(chrm_vl_mean7, **BndParm.HIRS_WD)

def pytest_assertion_pass(item, lineno: int, orig: str, expl: str):
    print(f"{item} passed. Explanation: {expl}")
