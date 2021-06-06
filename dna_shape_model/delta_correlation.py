import sys
from data_preprocess2 import preprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotnine as p9
from plotnine import ggplot, aes, stat_bin_2d, xlim, ylim

# get dictionary from text file


def main(argv=None):
    if argv is None:
        argv = sys.argv
        #input argszw
        file = argv[1]
        #e.g. data/41586_2020_3052_MOESM4_ESM.txt


    prep = preprocess(file)
    # if want mono-nucleotide sequences
    data = prep.one_hot_encode()
    # if want dinucleotide sequences
    #dict = prep.dinucleotide_encode()

    readout = data["readout"]
    fw_fasta = data["forward"]
    rc_fasta = data["reverse"]

    np.set_printoptions(threshold=sys.maxsize)

    # change from list to numpy array
    y = np.asarray(readout)
    
    df = pd.DataFrame({'x': y[:-6], 'y': y[6:]})
    print(df)
    
    p = (ggplot(data=df,
                   mapping=aes(x='x', y='y'))
         + stat_bin_2d(bins=150)
         + xlim(-2.5, 2.5)
         + ylim(-2.5, 2.5)
         )
    print(p)

    

if __name__ == "__main__":
    sys.exit(main())

