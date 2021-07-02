from bq import plot_dinucleotide_heatmap
from util import append_reverse_compliment
from shape import find_all_shape_values
from reader import DNASequenceReader
from constants import RL

if __name__ == '__main__':
    reader = DNASequenceReader()
    rl_df = reader.get_processed_data()[RL]
    rl_df = append_reverse_compliment(rl_df)
    plot_dinucleotide_heatmap(rl_df, 'rl')
