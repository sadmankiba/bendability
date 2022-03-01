import numpy as np

from motif.motifs import MotifsM30, MotifsM35, LEN_MOTIF
from util.reader import DNASequenceReader
from util.constants import YeastChrNumList
from util.util import roman_to_num

class TestMotifsM35:
    def test_motif40_score(self):
        motif_40 = "_GAAGAGC"
        seq = DNASequenceReader.read_yeast_genome_file(roman_to_num(YeastChrNumList[4]))
        match_pos = seq.find(motif_40[1:])
        mtf = MotifsM35()
        max_score_pos = np.where(mtf._running_score[40] > 14)[0][0]
        assert max_score_pos - LEN_MOTIF / 2 + 1 == match_pos




class TestMotifsM30:
    def test_plot_ranked_tf(self):
        motifs = MotifsM30()
        figpath = motifs.plot_ranked_tf()
        assert figpath.is_file()

    def test_ranked_tf(self):
        motifs = MotifsM30()
        tfdf = motifs.ranked_tf()
        assert set(tfdf.columns) == set(["tf", "contrib_score"])

    def test_sorted_contrib(self):
        assert MotifsM30().sorted_contrib()[:4] == [71, 114, 74, 108]
