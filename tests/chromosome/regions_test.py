import pandas as pd

from chromosome.chromosome import Chromosome
from chromosome.regions import Regions


class TestRegionsContain:
    def test_contains(self, chrm_vl_mean7: Chromosome):
        containers = pd.DataFrame({"start": [3, 7, 9], "end": [4, 12, 10]})
        rgns = Regions(chrm_vl_mean7, regions=containers)
        assert rgns._contains_loc([4, 11, 21, 3]).tolist() == [True, True, False]
