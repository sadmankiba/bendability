import pytest

from conformation.loops import Loops
from chromosome.chromosome import Chromosome


@pytest.fixture
def loops_vl():
    return Loops(Chromosome("VL"))
