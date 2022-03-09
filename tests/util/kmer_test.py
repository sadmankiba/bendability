from util.kmer import KMer
from chromosome.chromosome import Chromosome


class TestKMer:
    def test_count(self, chrm_vl_mean7: Chromosome):
        assert KMer.count("TA", chrm_vl_mean7.seqf(25, 32)) == 1
        assert KMer.count("CG", chrm_vl_mean7.seqf(1, 10)) == 1
        assert KMer.count("TA", chrm_vl_mean7.seqf([6, 25], [13, 32])) == [0, 1]
        assert KMer.count("CG", chrm_vl_mean7.seqf([1, 5], [9, 8])) == [1, 0]
