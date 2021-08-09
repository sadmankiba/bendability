from prediction import Prediction
from chromosome import Chromosome, YeastChrNumList
from loops import Loops

if __name__ == '__main__':
    for chr_id in YeastChrNumList:
        Loops(Chromosome(chr_id, Prediction(30))).plot_scatter_mean_c0_nuc_linker_individual_loop()