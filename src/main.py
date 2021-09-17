from constants import YeastChrNumList
from chromosome import Chromosome
from meanloops import MultiChrmMeanLoopsAggregator, MultiChrmMeanLoopsCollector
from prediction import Prediction
from tad import HicExplBoundaries, MultiChrmHicExplBoundariesCollector, MultiChrmHicExplBoundariesAggregator

if __name__ == '__main__':
    aggr = MultiChrmMeanLoopsAggregator(MultiChrmMeanLoopsCollector((Prediction(30), ('VI', 'VII'))))
    aggr.plot_c0_vs_loop_size()