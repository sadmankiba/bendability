import math
import random 
import string

def sorted_split(df, n=1000, n_bins=1, ascending=False):
    """
    Sort data according to C0 value 
    params:
        df: dataframe 
        n: Top n data to use 
        n_bins: Number of bins(dataframes) to split to 
        ascending: sort order
    returns:
        A list of dataframes.
    """
    sorted_df = df.sort_values(by=['C0'], ascending=ascending)[0:n]
    
    return [ sorted_df.iloc[start_pos : start_pos + math.ceil(n / n_bins)] 
                for start_pos in range(0, n, math.ceil(n / n_bins)) ]


def get_possible_seq(size):
    """ 
    Generates all possible nucleotide sequences of particular length
    
    returns:
        a list of sequences
    """
    
    possib_seq = ['']
    
    for _ in range(size):
        possib_seq = [ seq + nc for seq in possib_seq for nc in ['A', 'T', 'G', 'C'] ]
    
    return possib_seq


def find_occurence(seq_list, unit_size):
    """
    Find number of occurences of all possible nucleotide sequences of particular size in a list of sequences.
    param:
        seq_list: List of DNA sequences 
        unit_size: unit bp sequence size 
    returns:
        a dictionary mapping unit nucleotide sequence to number of occurences
    """
    possib_seq = get_possible_seq(unit_size)
    seq_occur_map = dict()
    
    for seq in possib_seq:
        seq_occur_map[seq] = 0  

    for whole_seq in seq_list:
        for i in range(len(whole_seq) - unit_size + 1):
            seq_occur_map[whole_seq[i:i+unit_size]] += 1

    return seq_occur_map


def gen_random_sequences(n):
    """Generates n 50 bp random DNA sequences"""
    seq_list = []
    for _ in range(n):
        seq = ''
        
        for _ in range(50):
            d = random.random()
            if d < 0.25:
                c = 'A'
            elif d < 0.5:
                c = 'T'
            elif d < 0.75:
                c = 'G'
            else:
                c = 'C'
            
            seq += c
        
        seq_list.append(seq)
    
    return seq_list


def get_random_string(length):
    
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

