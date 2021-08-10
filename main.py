from tad import MultiChrmHicExplBoundaries
from prediction import Prediction
from constants import YeastChrNumList
from chromosome import Chromosome
from genes import Genes

if __name__ == '__main__':
    MultiChrmHicExplBoundaries(Prediction(30), YeastChrNumList).plot_bar_perc_in_prmtrs() 