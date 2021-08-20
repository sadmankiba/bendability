from constants import CHRV_TOTAL_BP
from loops import Loops
from meanloops import MeanLoops, MultiChrmCoverLoopsCollector, MultiChrmMeanLoopsAggregator, MultiChrmMeanLoopsCollector
from chromosome import Chromosome

import pandas as pd
import numpy as np

from pathlib import Path

from prediction import Prediction


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

# TODO: Check loading tensorflow is slowing down unit tests 

class TestMultiChrmMeanLoopsCollector:
    def test_add_non_loop_mean(self):
        coll = MultiChrmMeanLoopsCollector(Prediction(30), ('VI', 'IV'))
        nl_col = coll._add_non_loop_mean()
        assert nl_col in coll._mcloop_df.columns 
        assert all(coll._mcloop_df[nl_col] > -0.5)
        assert all(coll._mcloop_df[nl_col] < 0)

    def test_add_num_loops_gt_non_loop(self):
        coll = MultiChrmMeanLoopsCollector(Prediction(30), ('VI', 'IV'))
        num_col = coll._add_num_loops()
        lt_col = coll._add_num_loops_lt_non_loop()
        assert lt_col in coll._mcloop_df.columns 
        assert all(coll._mcloop_df[lt_col] > coll._mcloop_df[num_col] * 0.3)
        assert all(coll._mcloop_df[lt_col] < coll._mcloop_df[num_col])
        
    def test_save_stat_all_methods(self):
        path = Path(MultiChrmMeanLoopsCollector(
            None, ('VL', )).save_avg_c0_stat())
        assert path.is_file()

        collector_df = pd.read_csv(path, sep='\t')
        assert np.isnan(collector_df.iloc[0].drop(['ChrID', 'model']).astype(float)).any() == False

    def test_get_loops_data(self):
        coll = MultiChrmMeanLoopsCollector(Prediction(30), ('VL', 'VII'))
        all_loops_df = coll.get_loops_data()
        assert len(all_loops_df) == len((coll._mcloops)[0]) + len((coll._mcloops)[1])
        
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

class TestMultiChrmMeanLoopsAggregator:
    def test_loop_l_lt_nll(self):
        aggr = MultiChrmMeanLoopsAggregator(MultiChrmMeanLoopsCollector(Prediction(30), ('X', 'XI')))
        aggr._loop_l_lt_nll()
        assert 'loop_l_lt_nll' in aggr._agg_df.columns.tolist()

    def test_save_stat(self):
        aggr = MultiChrmMeanLoopsAggregator(
                MultiChrmMeanLoopsCollector(
                    Prediction(30), ('XIII', 'III', 'IX')))
        
        path = aggr.save_stat([0])
        assert path.is_file()
    
    def test_plot_c0_vs_loop_size(self): 
        aggr = MultiChrmMeanLoopsAggregator(MultiChrmMeanLoopsCollector(Prediction(30), ('X', 'XI')))
        path = aggr.plot_c0_vs_loop_size()
        assert path.is_file()

class TestMultiChrmCoverLoopsCollector:
    def test_get_cover_stat(self):
        colt_df, path_str = MultiChrmCoverLoopsCollector(('VL',), 1000000).get_cover_stat()
        assert Path(path_str).is_file()
        assert pd.read_csv(path_str, sep='\t').columns.tolist() \
            == ['ChrID', 'loop_nuc', 'loop_linker', 'non_loop_nuc', 'non_loop_linker'] 

    def test_plot_cover_stat(self):
        path_str = MultiChrmCoverLoopsCollector(('VL',), 100000).plot_bar_cover_stat()
        assert Path(path_str).is_file()