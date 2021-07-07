from bq import plot_dinucleotide_heatmap
from util import append_reverse_compliment, HelicalSeparationCounter
from shape import find_all_shape_values
from reader import DNASequenceReader
from constants import RL

if __name__ == '__main__':
    HelicalSeparationCounter().count_dist_random_seq()