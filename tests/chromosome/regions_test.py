import pandas as pd

from chromosome.regions import contains


class TestRegions:
    def test_contains(self):
        containers = pd.DataFrame({"start": [3, 7, 9], "end": [4, 12, 10]})
        assert contains(containers, [4, 11, 21, 3]).tolist() == [True, True, False]
