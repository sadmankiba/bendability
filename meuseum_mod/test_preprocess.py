from data_preprocess import Preprocess
# Import from parent directory
import sys 
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from reader import CHRVL_FILE


class TestPreprocess:
    def test_get_sequences_target(self):
        prep = Preprocess(f'../data/input_data/bendability/{CHRVL_FILE}')
        seq_target = prep.get_sequences_target()
        assert set(seq_target.keys()) == set(['all_seqs', 'rc_seqs', 'target'])
