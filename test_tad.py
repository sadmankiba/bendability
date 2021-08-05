from tad import Boundary
from custom_types import YeastChrNum

class TestTad:
    def test_get_boundaries(self):
        boundaries = Boundary()._get_all_boundaries()
        assert len(boundaries) > 0
        chrm, region = boundaries[0].split(":")
        assert chrm in YeastChrNum
        resolution_start, resolution_end = region.split("-")
        assert int(resolution_start) >= 0
        assert int(resolution_end) >= 0
    
    def test_get_boundaries_in(self):
        regions = Boundary().get_boundaries_in('XIII')
        assert len(regions) > 0
        a_region = regions[0]
        assert a_region.score > 0
        assert a_region.chromosome == 'XIII'
        assert type(a_region.start) == int
        assert a_region.start > 0
        assert a_region.end > 0
        assert a_region.center > 0
