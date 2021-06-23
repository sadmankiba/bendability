from __future__ import annotations

import pandas as pd



class DNASequenceReader:
    """
    Reads and returns processed DNA sequence libraries
    """
    def __init__(self):
        pass 


    def _get_raw_data(self):
        CNL_FILE = 'data/41586_2020_3052_MOESM4_ESM.txt'
        cnl_df_raw = pd.read_table(CNL_FILE, sep='\t')
        print('cnl_df\n', cnl_df_raw)

        RL_FILE = 'data/41586_2020_3052_MOESM6_ESM.txt'
        rl_df_raw = pd.read_table(RL_FILE, sep='\t')
        print('rl_df\n', rl_df_raw)

        TL_FILE = 'data/41586_2020_3052_MOESM8_ESM.txt'
        tl_df_raw = pd.read_table(TL_FILE, sep='\t')
        print('tl_df\n', tl_df_raw)

        CHRV_FILE = 'data/41586_2020_3052_MOESM9_ESM.txt'
        chrvl_df_raw = pd.read_table(CHRV_FILE, sep='\t')
        print('chrvl_df\n', chrvl_df_raw)

        LIBL_FILE = 'data/41586_2020_3052_MOESM11_ESM.txt'
        libl_df_raw = pd.read_table(LIBL_FILE, sep='\t')
        print('libl_df\n', libl_df_raw)

        print(cnl_df_raw.keys())

        return (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw)


    def _preprocess(self, df, file_name):
        columns = ["Sequence #", "Sequence", " C0"]
        df = df[columns]
        df.columns = ["Sequence #", "Sequence", "C0"]
        
        for i in range(len(df)):
            df.at[i, 'Sequence'] = df['Sequence'][i][25:-25] 
        
        df.to_csv(f'data/{file_name}.csv', index=False)
        
        return df


    def get_processed_data(self) -> dict[str, pd.DataFrame]:
        """
        Get processed DNA sequence libraries

        returns :
            A dict mapping keys `['cnl', 'rl', 'tl', chrvl', libl']` to 
                pandas Dataframe with columns `["Sequence #", "Sequence", "C0"]`
            
        """ 
        (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw) = self._get_raw_data()

        cnl_df = self._preprocess(cnl_df_raw, 'cnl')
        rl_df = self._preprocess(rl_df_raw, 'rl')
        tl_df = self._preprocess(tl_df_raw, 'tl')
        chrvl_df = self._preprocess(chrvl_df_raw, 'chrvl')
        libl_df = self._preprocess(libl_df_raw, 'libl')

        return {
            'cnl': cnl_df, 
            'rl': rl_df, 
            'tl': tl_df, 
            'chrvl': chrvl_df, 
            'libl': libl_df
        }