from data_organizer import DataOrganizeOptions, DataOrganizer
from constants import CNL, TL, RL 

class Correlation:
    def kmer_corr(self, library_name: str):
        libraries = {
            'train': [library_name],
            'test': [], 
            'train_test': [],
            'seq_start_pos': 1,
            'seq_end_pos': 50
        }
        
        for k in [2, 3, 4]:
            options = DataOrganizeOptions(k_list=[k])

            organizer = DataOrganizer(libraries, None, None, options)
            kmer_df = organizer._get_kmer_count()['train'][0]
            kmer_df = kmer_df.drop(columns=['Sequence #', 'Sequence'])
            C0_corr = kmer_df.corr()['C0']
            C0_corr.sort_values(ascending=False).to_csv(
                f'data/generated_data/correlation/{library_name}_{k}_corr.tsv', sep='\t', \
                    index=True, float_format='%.2f')

    def hel_corr(self, library_name: str):
        libraries = {
            'train': [library_name],
            'test': [], 
            'train_test': [],
            'seq_start_pos': 1,
            'seq_end_pos': 50
        }

        organizer = DataOrganizer(libraries, None, None)
        hel_df = organizer._get_helical_sep()['train'][0]
        hel_df = hel_df.drop(columns=['Sequence #', 'Sequence'])
        hel_corr = hel_df.corr()['C0']
        hel_corr.sort_values(ascending=False).to_csv(
            f'data/generated_data/correlation/{library_name}_hel_corr.tsv', sep='\t', \
                index=True, float_format='%.2f'
        )