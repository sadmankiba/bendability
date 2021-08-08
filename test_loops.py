from constants import CHRV_TOTAL_BP
from loops import Loops, MeanLoops, MultiChrmCoverLoopsCollector, MultiChrmMeanLoopsCollector
from chromosome import Chromosome

import pandas as pd
import numpy as np

import unittest
import subprocess
from pathlib import Path

from prediction import Prediction


class TestLoops(unittest.TestCase):
    def test_read_loops(self):
        loop = Loops(Chromosome('VL', None))
        df = loop._read_loops()
        assert set(df.columns) == set(['start', 'end', 'res', 'len'])

        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop._loop_file])
        assert len(df) == int(s.split()[0]) - 2

    def test_exclude_above_len(self):
        bf_loops = Loops(Chromosome('VL', None))
        bf_len = len(bf_loops._loop_df)
        bf_arr = bf_loops.get_loop_cover(bf_loops._loop_df)

        af_loops = Loops(Chromosome('VL', None), 100000)
        af_len = len(af_loops._loop_df)
        af_arr = af_loops.get_loop_cover(af_loops._loop_df)
        
        assert bf_len >= af_len
        assert bf_arr.sum() >= af_arr.sum()

    def test_get_loop_cover(self):
        loops = Loops(Chromosome('VL', None))
        chrm_arr = loops.get_loop_cover(loops._loop_df)
        assert chrm_arr.shape == (CHRV_TOTAL_BP, )
        perc = chrm_arr.sum() / CHRV_TOTAL_BP * 100
        assert 10 < perc < 90

    def test_plot_mean_c0_across_loops(self):
        chr = Chromosome('VL', None)
        loop = Loops(chr)
        loop.plot_mean_c0_across_loops(150)
        path = Path(f'figures/loop/mean_c0_p_150_mxl_100000_{chr}.png')
        assert path.is_file()

    def test_plot_c0_in_individual_loop(self):
        loop = Loops(Chromosome('VL', None))
        loop.plot_c0_in_individual_loop()
        assert Path(f'figures/loop/VL').is_dir()

    def test_plot_c0_around_anchor(self):
        chr = Chromosome('VL', None)
        loop = Loops(chr)
        loop.plot_c0_around_anchor(500)
        path = Path(f'figures/loop_anchor/dist_500_{chr}.png')
        assert path.is_file()

    def test_plot_nuc_across_loops(self):
        chr = Chromosome('II', Prediction(30))
        loop = Loops(chr)
        loop.plot_mean_nuc_occupancy_across_loops()
        path = Path(f'figures/loop/mean_nuc_occ_p_150_mxl_100000_{chr}.png')
        assert path.is_file()


class TestMeanLoops:
    def test_in_complete_loop(self):
        mean = MeanLoops(Loops(Chromosome('VL', None))).in_complete_loop()
        assert -0.3 < mean < 0

    def test_in_complete_non_loop(self):
        mean = MeanLoops(Loops(Chromosome('VL', None))).in_complete_non_loop()
        assert -0.4 < mean < 0

    def test_validate_complete_loop_non_loop(self):
        chrm = Chromosome('VL', None)
        chrm_mean = chrm.get_spread().mean()
        
        mloops = MeanLoops(Loops(chrm))
        loop_mean = mloops.in_complete_loop()
        non_loop_mean = mloops.in_complete_non_loop()
        assert (loop_mean < chrm_mean < non_loop_mean) or \
            (loop_mean > chrm_mean > non_loop_mean)

    def test_in_quartile_by_pos(self):
        arr = MeanLoops(Loops(Chromosome('VL',
                                   None))).in_quartile_by_pos()
        print(arr)
        assert arr.shape == (4, )

    def test_around_anc(self):
        avg = MeanLoops(Loops(Chromosome('VL',
                                   None))).around_anc('start', 500)
        assert avg > -1
        assert avg < 1

    def test_in_nuc_linker(self):
        mloops = MeanLoops(Loops(Chromosome('VL', None)))
        na, la = mloops.in_nuc_linker()
        assert -1 < na < 0
        assert -1 < la < 0
        assert na > la


class TestMultiChrmMeanLoopsCollector:
    def test_save_stat_all_methods(self):
        path = Path(MultiChrmMeanLoopsCollector(
            None, ('VL', )).save_avg_c0_stat())
        assert path.is_file()

        collector_df = pd.read_csv(path, sep='\t')
        assert np.isnan(collector_df.iloc[0].drop(['ChrID', 'model']).astype(float)).any() == False

    def test_save_stat_partial_call(self):
        path = Path(MultiChrmMeanLoopsCollector(None, ('VL', )).save_avg_c0_stat([0, 1, 3, 5],
                                                               True))
        assert path.is_file()

        collector_df = pd.read_csv(path, sep='\t')
        cols = ['chromosome', 'chrm_nuc', 'chrm_linker', 'loop', 'non_loop']
        assert np.isnan(collector_df[cols].iloc[0]).any() == False   

    def test_plot_loop_cover_frac(self):
        MultiChrmMeanLoopsCollector(None, ('VL', )).plot_loop_cover_frac()
        fig_path = Path('figures/mcloop/loop_cover_md_None_mx_None.png')
        assert fig_path.is_file()

class TestMultiChrmCoverLoopsCollector:
    def test_get_cover_stat(self):
        colt_df, path_str = MultiChrmCoverLoopsCollector(('VL',), 1000000).get_cover_stat()
        assert Path(path_str).is_file()
        assert pd.read_csv(path_str, sep='\t').columns.tolist() \
            == ['ChrID', 'loop_nuc', 'loop_linker', 'non_loop_nuc', 'non_loop_linker'] 

    def test_plot_cover_stat(self):
        path_str = MultiChrmCoverLoopsCollector(('VL',), 100000).plot_cover_stat()
        assert Path(path_str).is_file()