
from evaluation import Evaluation

# Import from parent directory
import sys 
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 

from constants import CHRVL

from pathlib import Path

class TestEvaluation:
    def test_check_performance(self):
        Evaluation().check_performance(CHRVL)
        assert Path(f'predictions/{CHRVL}_pred.tsv').is_file()