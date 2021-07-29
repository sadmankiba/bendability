from __future__ import annotations



# Import from parent directory
import sys 
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 

from reader import DNASequenceReader
from custom_types import LIBRARY_NAMES
from meuseum_mod.dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode

import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

import re
from typing import Literal


class Preprocess:
    def __init__(self, df: pd.DataFrame):
        """
        Construct a Preprocess object. 
        """
        self._df = df

    
    def _rc_comp2(self, seqn):
        '''
        Find reverse complement
        '''
        def reverse_compliment_of(seq: str):
            # Define replacements
            rep = {"A": "T", "T": "A", 'G': 'C', 'C': 'G'} 
            
            # Create regex pattern
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            
            # Replace and return reverse sequence 
            return (pattern.sub(lambda m: rep[re.escape(m.group(0))], seq))[::-1]

        all_sequences = []
        for seq in range(len(seqn)):
            all_sequences.append(reverse_compliment_of(seqn[seq]))

        return all_sequences


    def _get_sequences_target(self) -> dict[Literal['all_seqs', 'target', 'rc_seqs'], list]:
        all_seqs = self._df['Sequence'].tolist()
        rc_seqs = self._rc_comp2(all_seqs)
        
        # Set target
        target = self._df['C0'].tolist() if 'C0' in self._df else None  

        return {
            "all_seqs": all_seqs,
            "target": target,
            "rc_seqs": rc_seqs
        }


    def one_hot_encode(self) -> dict[str, np.ndarray]:
        """
        Encodes DNA sequences with one hot encoding

        Each sequence is transformed into a 2D-array of shape (50, 4)

        Returns:
            A dictionary containing three keys: ['forward', 'reverse', 'target']
        """
        # The LabelEncoder encodes a sequence of bases as a sequence of
        # integers.
        integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        # each row corresponds to one possible value of each feature.
        one_hot_encoder = OneHotEncoder(categories='auto')

        #dict = self.without_augment()
        seq_and_target = self._get_sequences_target()
        result = dict()
        result["target"] = np.asarray(seq_and_target['target'])

        forward = []
        reverse = []

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in seq_and_target["all_seqs"]:
            new_seq = 'ACGT' + sequence
            temp_seqs.append(new_seq)

        for sequence in temp_seqs:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            forward.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        lengths = []
        for i in range(len(forward)):
            length = len(forward[i])
            lengths.append(length)
        max_length = max(lengths)  # get the maxmimum length of all sequences

        for i in range(len(forward)):
            while (len(forward[i]) < max_length):
                forward[i] = np.vstack((forward[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        features = []
        for sequence in forward:
            new = sequence[4:]
            features.append(new)

        features = np.stack(features)
        result["forward"] = np.asarray(features)
        assert result["forward"][0].shape == (50, 4)

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in seq_and_target["rc_seqs"]:
            new_seq = 'ACGT' + sequence
            temp_seqs.append(new_seq)

        for sequence in temp_seqs:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            reverse.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        lengths = []
        for i in range(len(reverse)):
            length = len(reverse[i])
            lengths.append(length)
        max_length = max(lengths)  # get the maxmimum length of all sequences

        for i in range(len(reverse)):
            while (len(reverse[i]) < max_length):
                reverse[i] = np.vstack((reverse[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        features = []
        for sequence in reverse:
            new = sequence[4:]
            features.append(new)

        features = np.stack(features)
        result["reverse"] = np.asarray(features)
        assert result["reverse"][0].shape == (50, 4)

        return result


    def dinucleotide_encode(self):
        # Requires fix
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        forward_sequences = mono_to_dinucleotide(new_fasta)
        reverse_sequences = mono_to_dinucleotide(rc_fasta)

        forward = dinucleotide_one_hot_encode(forward_sequences)
        reverse = dinucleotide_one_hot_encode(reverse_sequences)

        dict = {}
        dict["readout"] = self.read_readout()
        dict["forward"] = forward
        dict["reverse"] = reverse
        return dict
