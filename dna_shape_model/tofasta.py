import time
import sys
import numpy as np
from data_preprocess2 import preprocess
import pandas as pd

def main(argv = None):
    if argv is None:
        argv = sys.argv
        file = argv[1]
        #e.g. data/41586_2020_3052_MOESM4_ESM.txt
        out = argv[2]
        #e.g. data/sequences.fasta
        direction = argv[3]

    ## excute the code
    start_time = time.time()
    #Reproducibility

    prep = preprocess(file)
    # if want mono-nucleotide sequences
    if direction == '-f':
        fasta = prep.read_fasta_forward()
    elif direction == '-r':
        fasta = prep.rc_comp2()

    np.set_printoptions(threshold=sys.maxsize)
    df = pd.DataFrame({'sequences': fasta,})
    df.to_csv(out, sep='\t', index=False, header=False)

        
    # reports time consumed during execution (secs)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    sys.exit(main())


