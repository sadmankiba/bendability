from motif.motifs import Motifs

class TestMotifs:
    def test_plot_ranked_tf(self):
        motifs = Motifs()
        figpath = motifs.plot_ranked_tf()
        assert figpath.is_file()

    def test_ranked_tf(self):
        motifs = Motifs()
        tfdf = motifs.ranked_tf()
        assert set(tfdf.columns) == set(["tf", "contrib_score"])
    
    def test_sorted_contrib(self):
        assert Motifs().sorted_contrib()[:4] == [71, 114, 74, 108]
