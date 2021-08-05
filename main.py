from prediction import Prediction
from chromosome import Chromosome
from nucleosome import Nucleosome

import itertools

if __name__ == '__main__':
    for chr_id, nuc_half  in itertools.product(['VL', 'X'], [73, 50]): 
        (na, la) = Nucleosome(Chromosome(chr_id, Prediction(30))).find_avg_nuc_linker_c0(nuc_half)
        print(na, la)
