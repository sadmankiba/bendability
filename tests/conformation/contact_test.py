from chromosome.chromosome import Chromosome
from conformation.contact import Contact

class TestContact:
    def test_load(self):
        contact = Contact(Chromosome('VL'))
        df = contact._load_contact()
        assert df.columns.tolist() == ['row', 'col', 'intensity']
        print(len(df))
        assert len(df) > 0