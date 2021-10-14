import pandas as pd 

from chromosome.chromosome import Chromosome
from util.util import PathUtil

# TODO: Rename conformation to threedim

class Contact:
    def __init__(self, chrm: Chromosome):
        self._chrm = chrm 
        self._matrix = self._load_contact()

    def _load_contact(self) -> pd.DataFrame:
        return pd.read_table(f'{PathUtil.get_data_dir()}/input_data/contact/'
            f'observed_vc_400_{self._chrm.number}.txt', names=['row', 'col', 'intensity'])
        