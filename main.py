from occurence import Occurence
from util import append_reverse_compliment, HelicalSeparationCounter
from shape import find_all_shape_values
from reader import DNASequenceReader
from constants import RL

if __name__ == '__main__':
    reader = DNASequenceReader()
    all_lib = reader.get_processed_data()
    lib = RL
    df = append_reverse_compliment(all_lib[RL])
    Occurence().plot_boxplot(df, lib)