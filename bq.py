from util import gen_random_sequences, sorted_split, find_occurence, get_possible_seq

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def find_bq(df, unit_size):
    """
    Find bendability quotient of all possible neucleotide sequences of particular size

    returns:
        A dictionary mapping neucleotide sequences to bendability quotient

    """
    N_BINS = 12
    
    # get sorted split bins of equal size
    n_seq = len(df) - len(df) % N_BINS
    df = df.iloc[:n_seq, :]
    sorted_dfs = sorted_split(df, n=len(df), n_bins=N_BINS, ascending=True)
    
    # For each bin, find occurence of bp sequences 
    possib_seq = get_possible_seq(unit_size)
    
    seq_occur_map = dict()  # mapping of possib_seq to a list of occurences

    for seq in possib_seq:
        seq_occur_map[seq] = np.array([]) 

    for sdf in sorted_dfs:
        one_bin_occur_dict = find_occurence(sdf['Sequence'].tolist(), unit_size)
        for unit_seq in seq_occur_map:
            seq_occur_map[unit_seq] = np.append(seq_occur_map[unit_seq], one_bin_occur_dict[unit_seq])


    # Generate a large random list of 50 bp DNA sequences
    # and find avg. occurences for a bin
    GENERATE_TIMES = 100
    num_random_sequences = len(sorted_dfs[0]) * GENERATE_TIMES
    random_seq_list = gen_random_sequences(num_random_sequences)
    random_list_occur_dict = find_occurence(random_seq_list, unit_size)

    for unit_seq in random_list_occur_dict:
        random_list_occur_dict[unit_seq] /= GENERATE_TIMES

    # Normalize 
    for unit_seq in seq_occur_map:
        seq_occur_map[unit_seq] = \
            seq_occur_map[unit_seq] / random_list_occur_dict[unit_seq]

    print('normalized sequence occur map\n', seq_occur_map)

    # Find mean C0 value in each bin 
    mean_c0 = [ np.mean(sdf['C0']) for sdf in sorted_dfs ]
    print('mean_c0\n', mean_c0)
    
    # Fit straight line on C0 vs. normalized occurence
    bq_map = dict()
    for unit_seq in seq_occur_map:
        X = np.reshape(np.array(mean_c0), (-1, 1))
        y = seq_occur_map[unit_seq]
        lr = LinearRegression().fit(X, y)
        bq_map[unit_seq] = lr.coef_[0]

    print('bq_map\n', bq_map)
    return bq_map