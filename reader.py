from __future__ import annotations

from constants import CNL, RL, TL, CHRVL, LIBL

import pandas as pd


class DNASequenceReader:
    """
    Reads and returns processed DNA sequence libraries
    """
    def __init__(self):
        self._bendability_data_dir='data/input_data/bendability'
        self._nuc_center_file = 'data/input_data/nucleosome_position/41586_2012_BFnature11142_MOESM263_ESM.txt'


    def _get_raw_data(self):
        CNL_FILE = f'{self._bendability_data_dir}/41586_2020_3052_MOESM4_ESM.txt'
        cnl_df_raw = pd.read_table(CNL_FILE, sep='\t')

        RL_FILE = f'{self._bendability_data_dir}/41586_2020_3052_MOESM6_ESM.txt'
        rl_df_raw = pd.read_table(RL_FILE, sep='\t')

        TL_FILE = f'{self._bendability_data_dir}/41586_2020_3052_MOESM8_ESM.txt'
        tl_df_raw = pd.read_table(TL_FILE, sep='\t')

        CHRV_FILE = f'{self._bendability_data_dir}/41586_2020_3052_MOESM9_ESM.txt'
        chrvl_df_raw = pd.read_table(CHRV_FILE, sep='\t')

        LIBL_FILE = f'{self._bendability_data_dir}/41586_2020_3052_MOESM11_ESM.txt'
        libl_df_raw = pd.read_table(LIBL_FILE, sep='\t')

        return (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw)


    def _preprocess(self, df: pd.DataFrame):
        df = df[["Sequence #", "Sequence", " C0"]].rename(columns={" C0": "C0"})
        df['Sequence'] = df['Sequence'].str[25:-25]
        
        return df


    def get_processed_data(self) -> dict[str, pd.DataFrame]:
        """
        Get processed DNA sequence libraries

        returns :
            A dict mapping library names to 
                pandas Dataframe with columns `["Sequence #", "Sequence", "C0"]`
            
        """ 
        (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw) = self._get_raw_data()
        
        cnl_df = self._preprocess(cnl_df_raw)
        rl_df = self._preprocess(rl_df_raw)
        tl_df = self._preprocess(tl_df_raw)
        chrvl_df = self._preprocess(chrvl_df_raw)
        libl_df = self._preprocess(libl_df_raw)

        return {
            CNL: cnl_df, 
            RL: rl_df, 
            TL: tl_df, 
            CHRVL: chrvl_df, 
            LIBL: libl_df
        }


    def read_library_prediction(self, lib_name: str):
        """Read predicted C0 by meuseum model"""

        predict_df = pd.read_table(f'meuseum_mod/predictions/{lib_name}_pred.csv', sep='\t')
        predict_df = predict_df.assign(seq_no = lambda df: df.index + 1)\
                        .rename(columns = {
                            'seq_no': 'Sequence #',
                            'Predicted Value': 'C0' 
                        }).drop(columns=['True Value'])
        
        return predict_df 


    def read_nuc_center(self) -> pd.DataFrame:
        """
        Read nucleosome center position data. 
        
        Data is provided by paper "A map of nucleosome positions in yeast at base-pair resolution"

        Returns:
            A `pandas.DataFrame` with columns ['Chromosome ID', 'Position', 'NCP score', 'NCP score/noise']
        """
        return pd.read_table(self._nuc_center_file, 
                    delim_whitespace=True,
                    header=None, 
                    names=['Chromosome ID', 'Position', 'NCP score', 'NCP score/noise']
                )