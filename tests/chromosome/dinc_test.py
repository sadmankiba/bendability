from typing import Callable
import pytest

from chromosome.dinc import Dinc


@pytest.fixture
def create_dinc():
    def _create_dinc(seq: str) -> Dinc:
        dinc = Dinc(None)

        class Dummy:
            pass

        dinc._chrm = Dummy()
        dinc._chrm.seq = seq
        return dinc

    return _create_dinc


CreateDincT = Callable[[str], Dinc]


class TestDinc:
    def test_dinc_count_multisegment(self, create_dinc: CreateDincT):
        dinc = create_dinc("GTCGGTATCGCTA")
        assert dinc.ta_count_multisegment([6, 2], [13, 10]) == [2, 1]
        assert dinc.cg_count_multisegment([1, 5], [9, 8]) == [1, 0]

    def test_dinc_count_singlesegment(self, create_dinc: CreateDincT):
        dinc = create_dinc("GTCGGTATCGCTA")
        assert len(dinc._chrm.seq) == 13
        assert dinc.ta_count_singlesegment(6, 13) == 2
        assert dinc.cg_count_singlesegment(1, 10) == 2
