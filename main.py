from occurence import Occurence
from util import append_reverse_compliment
from shape import find_all_shape_values
from reader import DNASequenceReader
from constants import RL, CNL, TL
from correlation import Correlation


def plotting_boxplot():
    reader = DNASequenceReader()
    all_lib = reader.get_processed_data()
    lib = RL
    df = append_reverse_compliment(all_lib[RL])
    Occurence().plot_boxplot(df, lib)


if __name__ == '__main__':
    corr = Correlation()
    corr.kmer_corr(RL)
    corr.kmer_corr(CNL)
    corr.kmer_corr(TL)