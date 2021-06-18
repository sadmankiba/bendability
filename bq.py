from util import gen_random_sequences, sorted_split, find_occurence, get_possible_seq
from read_data import get_processed_data
from shape import run_dna_shape_r_wrapper, SHAPE_FULL_FORM

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pathlib import Path


def find_bq(df, unit_size, df_str):
    """
    Find bendability quotient of all possible neucleotide sequences of particular size

    args:
        df `(pandas.DataFrame)`: A dataframe with columns Sequence #, Sequence, C0
        unit_size `(int)`: Size of unit sequence bp 
    returns:
        A dictionary mapping neucleotide sequences to bendability quotient

    """
    N_BINS = 12
    bq_file = Path(f'data/generated_data/bq/{df_str}_bq_{unit_size}.tsv')
    
    # If bq value is already stored, return it 
    if bq_file.is_file():
        with open(bq_file, 'r') as f:
            bq_df = pd.read_csv(f, sep='\t')
        return bq_df.set_index('Sequence').to_dict()['BQ']
         
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

    # Save BQ value
    bq_seq_val_pairs = bq_map.items()
    bq_df = pd.DataFrame({
                'Sequence': [ pair[0] for pair in bq_seq_val_pairs ], 
                'BQ': [ pair[1] for pair in bq_seq_val_pairs ]
            })
    bq_df.sort_values('BQ', ignore_index=True, inplace=True)
    with open(bq_file, 'w') as f:
        bq_df.to_csv(f, sep='\t', index=False)

    return bq_map

def plot_bq():
    UNIT_NUCLEOTIDE_SIZE = 5

    (cnl_df, rl_df, tl_df, chrvl_df, libl_df) = get_processed_data()

    seq_bq_map = find_bq(cnl_df, unit_size=UNIT_NUCLEOTIDE_SIZE)

    sorted_unit_seq_bq_pair = sorted(seq_bq_map.items(), key=lambda x: x[1])
    print('sorted_unit_seq_bq_pair\n', sorted_unit_seq_bq_pair)

    # Plot
    if len(sorted_unit_seq_bq_pair) > 20:
        pairs_to_show = [ pair for (i, pair) in enumerate(sorted_unit_seq_bq_pair) if i < 10 or i >= len(sorted_unit_seq_bq_pair) - 10 ]
    else:
        pairs_to_show = sorted_unit_seq_bq_pair

    print('pairs to show\n', pairs_to_show)

    x = [ pair[0] for pair in pairs_to_show]
    y = [ pair[1] for pair in pairs_to_show]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Move bottim x-axis to centre
    ax.spines['bottom'].set_position(('data', 0))

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.bar(x, y)
    plt.ylabel('Bendability quotient')
    plt.setp(ax.get_xticklabels(), rotation=90, va='top')
    plt.savefig('figures/cnl_bq_5.png')

    plt.show()
    
def plot_shape_seq_bq():
    TOP_N = 20      # Number of top or bottom sequences to consider

    (cnl_df, rl_df, tl_df, chrvl_df, libl_df) = get_processed_data()

    for df_name, df in [('cnl', cnl_df), ('rl', rl_df)]:
        for unit_bp_size in [6, 7]: 
            seq_bq_map = find_bq(df, unit_bp_size, df_name)

            sorted_unit_seq_bq_pair = sorted(seq_bq_map.items(), key=lambda x: x[1])
            print('sorted_unit_seq_bq_pair - no. of keys', len(sorted_unit_seq_bq_pair))
            assert len(sorted_unit_seq_bq_pair[0][0]) == unit_bp_size

            # Choose top n unit sequences at both extreme
            pairs_to_show = [ pair for (i, pair) in enumerate(sorted_unit_seq_bq_pair) if i < TOP_N or i >= len(sorted_unit_seq_bq_pair) - TOP_N]
            
            x = [pair[0] for pair in pairs_to_show]
            assert len(x) == TOP_N * 2

            seq_df = pd.DataFrame({'Sequence': x})
            shape_arr_map = run_dna_shape_r_wrapper(seq_df, True)
            
            
            for shape_name in ['HelT', 'MGW', 'ProT', "Roll"]:
                plt.close()
                plt.clf()
                plt.figure(clear=True)
                shape_arr = shape_arr_map[shape_name]
                if shape_name in ['HelT', 'Roll']:
                    shape_arr = shape_arr[:, 1:-1] 
                assert (shape_arr.shape[1] == unit_bp_size - (4 if shape_name in ['MGW', 'ProT'] else 5)), \
                    'Number of shapes: ' + str(shape_arr.shape[1]) + ' is not okay' 
                    
                for i in range(shape_arr.shape[0]):
                    color = 'b' if i < TOP_N else 'r'
                    plt.plot(range(shape_arr.shape[1]), shape_arr[i], linestyle='-', marker='o', color=color)

                blue_patch = mpatches.Patch(color='r', label=f'Top {TOP_N} {shape_arr.shape[1]}-shape sequences associated with high loopability')
                green_patch = mpatches.Patch(color='b', label=f'Top {TOP_N} {shape_arr.shape[1]}-shape sequences associated with low loopability')
                ax = plt.gca()
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), handles=[blue_patch, green_patch])
                plt.xlabel('Position')
                plt.xticks(ticks=np.arange(shape_arr.shape[1]), labels= [str(i+1) for i in range(shape_arr.shape[1])])
                plt.ylabel(SHAPE_FULL_FORM[shape_name])
                 
                plt.savefig(f'figures/shape_seq/{df_name}_{shape_name}_{shape_arr.shape[1]}_{TOP_N}_seq_impact.png')
                
                

