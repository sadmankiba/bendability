import pytest 

from chromosome.chromosome import Chromosome


@pytest.fixture
def chrm_vl():
    return Chromosome('VL')

@pytest.fixture
def chrm_i():
    return Chromosome('I')

def pytest_assertion_pass(item, lineno: int, orig: str, expl: str):
    print(f"{item} passed. Explanation: {expl}")
