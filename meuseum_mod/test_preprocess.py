
from data_preprocess import Preprocess
# Import from parent directory
import sys 
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from constants import CHRVL, CHRVL_LEN, RL, RL_LEN

class TestPreprocess:
    def test_get_sequences_target(self):
        prep = Preprocess(CHRVL)
        seq_target = prep.get_sequences_target()
        assert set(seq_target.keys()) == set(['all_seqs', 'rc_seqs', 'target'])
        assert len(seq_target['all_seqs']) == CHRVL_LEN
        assert len(seq_target['rc_seqs']) == CHRVL_LEN
        assert len(seq_target['target']) == CHRVL_LEN
    
    def test_one_hot_encode(self):
        prep = Preprocess(RL)
        data = prep.one_hot_encode()
        assert data['forward'].shape == (RL_LEN, 50, 4)
        assert data['reverse'].shape == (RL_LEN, 50, 4)
        