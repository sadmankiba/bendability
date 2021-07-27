from data_preprocess import Preprocess
from museum_constants import LIBRARIES

class TestPreprocess:
    def test_get_sequences_target(self):
        lib = LIBRARIES['chrvl']
        prep = Preprocess(f'../data/input_data/bendability/{lib["file"]}')
        seq_target = prep.get_sequences_target()
        assert set(seq_target.keys()) == set(['all_seqs', 'rc_seqs', 'target'])
