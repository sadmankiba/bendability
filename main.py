from chromosome import Chromosome
from data_organizer import DataOrganizer, ShapeOrganizerFactory, \
        SequenceLibrary, DataOrganizeOptions, TrainTestSequenceLibraries
from feat_selector import FeatureSelectorFactory
from occurence import Occurence
from util import append_reverse_compliment
from reader import DNASequenceReader
from constants import CHRV_TOTAL_BP, RL, TL
from model import Model
from loops import Loops, MultiChrLoops

import numpy as np

def plotting_boxplot():
    reader = DNASequenceReader()
    all_lib = reader.get_processed_data()
    lib = RL
    df = append_reverse_compliment(all_lib[RL])
    Occurence().plot_boxplot(df, lib)


def run_model():
    libraries: TrainTestSequenceLibraries = {
        'train': [ SequenceLibrary(name=TL, quantity=20000) ],
        'test': [ SequenceLibrary(name=RL, quantity=5000) ], 
        'train_test': [],
        'seq_start_pos': 1,
        'seq_end_pos': 50
    }

    # shape_factory = ShapeOrganizerFactory('normal', 'ProT')
    # shape_organizer = shape_factory.make_shape_organizer(library)
    feature_factory = FeatureSelectorFactory('corr')
    selector = feature_factory.make_feature_selector()

    options: DataOrganizeOptions = {
        'k_list': [2],
        'range_split': np.array([0.2, 0.6, 0.2]),
        'binary_class': False,
        'balance': False,
        'c0_scale': 20
    }

    organizer = DataOrganizer(libraries, None, selector, options)

    model = Model(organizer)

    # model.run_seq_classifier()
    model.run_seq_regression()
    # model.run_shape_cnn_classifier()


if __name__ == '__main__':
    MultiChrLoops().find_avg_c0()