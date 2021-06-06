import sys
import numpy as np
import pandas as pd


class preprocess:
    def __init__(self, file, rc_file, readout_file):
        self.file = file
        self.rc_file = rc_file
        self.readout_file = readout_file
    
    # to augment on readout data
    def read_readout(self):
        df = pd.read_table(filepath_or_buffer=self.readout_file)
        all_readouts = []
        for r in df[' C0']:
            all_readouts.append(r)
        return all_readouts

    def augment(self):
        forward = self.stack(self.file)
        reverse = self.stack(self.rc_file)
        readout = self.read_readout()

        dict = {
            "forward": forward,
            "readout": readout,
            "reverse": reverse}
        return dict

    def stack(self, file):
        self.df_ep = pd.read_table(filepath_or_buffer=file+'.EP.pre', header=None, sep=',')
        self.df_helt = pd.read_table(filepath_or_buffer=file+'.HelT.pre', header=None, sep=',')
        self.df_mgw = pd.read_table(filepath_or_buffer=file+'.MGW.pre', header=None, sep=',')
        self.df_prot = pd.read_table(filepath_or_buffer=file+'.ProT.pre', header=None, sep=',')
        self.df_roll = pd.read_table(filepath_or_buffer=file+'.Roll.pre', header=None, sep=',')
        
        print(self.df_ep.head())
        print(self.df_helt.head())
        print(self.df_mgw.head())
        print(self.df_prot.head())
        print(self.df_roll.head())

        self.df_ep[46] = 0
        self.df_mgw[46] = 0
        self.df_prot[46] = 0

        f_ep = self.df_ep.to_numpy()
        f_helt = self.df_helt.to_numpy()
        f_mgw = self.df_mgw.to_numpy()
        f_prot = self.df_prot.to_numpy()
        f_roll = self.df_roll.to_numpy()

        sequences = np.swapaxes(np.stack((f_ep, f_helt, f_mgw, f_prot, f_roll), axis=1), 1, 2).astype(np.float32)

        return sequences
