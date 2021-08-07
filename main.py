from prediction import Prediction
from chromosome import ChrIdList, Chromosome, YeastChrNumList
from nucleosome import Nucleosome
from loops import Loops, MultiChrmMeanLoopsCollector

import itertools

if __name__ == '__main__':
    MultiChrmMeanLoopsCollector(6, [ChrIdList]).plot_loop_nuc_linker_mean()
        
