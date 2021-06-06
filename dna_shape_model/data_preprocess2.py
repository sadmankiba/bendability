import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode
import pandas


class preprocess:
    def __init__(self, file):
        self.file = file
        self.df = pandas.read_table(filepath_or_buffer=file, )
        # self.read_fasta_into_list()
        # self.read_fasta_forward()
        # self.rc_comp2()
        # self.read_readout()
        # self.without_augment()
        # self.augment()
        # self.one_hot_encode()

    def read_fasta_into_list(self):
        all_seqs = []
        for s in self.df['Sequence']:
            all_seqs.append(s[25:-25])
        return all_seqs

    def read_fasta_forward(self):
        return self.read_fasta_into_list()

    # augment the samples with reverse complement
    def rc_comp2(self):

        def rc_comp(seq):
            rc_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
            rc_seq = ''.join([rc_dict[c] for c in seq[::-1]])
            return rc_seq

        seqn = self.read_fasta_into_list()
        all_sequences = []
        for seq in range(len(seqn)):
            all_sequences.append(rc_comp(seqn[seq]))

        # return all_sequences

        return all_sequences

    # to augment on readout data
    def read_readout(self):
        all_readouts = []
        for r in self.df[' C0']:
            all_readouts.append(r)
        return all_readouts

    def augment(self):
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        readout = self.read_readout()

        dict = {
            "new_fasta": new_fasta,
            "readout": readout,
            "rc_fasta": rc_fasta}
        return dict

    def without_augment(self):
        new_fasta = self.read_fasta_into_list()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout": readout}
        return dict

    def one_hot_encode(self):
        # The LabelEncoder encodes a sequence of bases as a sequence of
        # integers.
        integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        # each row corresponds to one possible value of each feature.
        one_hot_encoder = OneHotEncoder(categories='auto')

        #dict = self.without_augment()
        dict = self.augment()

        forward = []
        reverse = []

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in dict["new_fasta"]:
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
        dict["forward"] = features

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in dict["rc_fasta"]:
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
        dict["reverse"] = features

        return dict

    def dinucleotide_encode(self):
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
