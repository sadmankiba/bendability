import pytest

from chromosome.chromosome import Chromosome
from conformation.loops import Loops


@pytest.fixture
def loops_vl(chrm_vl):
    return Loops(chrm_vl)

@pytest.fixture
def chrm_vl_mean7():
    return Chromosome("VL", spread_str="mean7")