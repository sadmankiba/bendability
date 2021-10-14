import numpy as np

from chromosome.chromosome import Chromosome
from conformation.contact import Contact

class TestContact:
    # TODO: Use fixture for Chromosome('VL')
    def test_generate_matrix(self):
        contact = Contact(Chromosome('VL'))
        mat = contact._generate_mat()
        assert np.count_nonzero(mat) > 0.08 * mat.size
        assert np.count_nonzero(mat.diagonal()) > 0.9 * mat.diagonal().size

    def test_load(self):
        contact = Contact(Chromosome('VL'))
        df = contact._load_contact()
        assert df.columns.tolist() == ['row', 'col', 'intensity']
        print(len(df))
        assert len(df) > 0