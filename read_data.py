import pandas as pd

def get_raw_data():
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
    
def preprocess(df, file_name):
    columns = ["Sequence #", "Sequence", " C0"]
    df = df[columns]
    df.columns = ["Sequence #", "Sequence", "C0"]
    
    for i in range(len(df)):
        df.at[i, 'Sequence'] = df['Sequence'][i][25:-25] 
    
    df.to_csv(f'data/{file_name}.csv', index=False)
    
    return df

def get_processed_data():
    """
    returns :
        sequence library as pandas Dataframe with columns ["Sequence #", "Sequence", "C0"]
        
    """ 
    (cnl_df_raw, rl_df_raw, tl_df_raw, chrvl_df_raw, libl_df_raw) = get_raw_data()

    cnl_df = preprocess(cnl_df_raw, 'cnl')
    rl_df = preprocess(rl_df_raw, 'rl')
    tl_df = preprocess(tl_df_raw, 'tl')
    chrvl_df = preprocess(chrvl_df_raw, 'chrvl')
    libl_df = preprocess(libl_df_raw, 'libl')

    return (cnl_df, rl_df, tl_df, chrvl_df, libl_df)