from prediction import Prediction
from chromosome import Chromosome
from nucleosome import Nucleosome
from loops import Loops

import itertools

if __name__ == '__main__':
    for chr_id, nuc_half  in itertools.product(['VL', 'X', 'III'], [73, 50]): 
        chrm = Chromosome(chr_id, Prediction(30))
        (na, la) = Nucleosome(chrm).find_avg_nuc_linker_c0(nuc_half)
        print(chr_id, nuc_half)
        print('In chromosome:', na, la)
        print('In loop:', Loops(chrm).find_mean_c0_in_nuc_linker(nuc_half))
        
