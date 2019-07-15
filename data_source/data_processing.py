import pandas as pd
import numpy as np
import os
current_path = os.path.dirname(__file__)




def is_tpa(df_case, keep_cols=False):
    df_tpa = df_case[['ICD_ID', 'IVTPATH_ID', 'IVTPA_DT', 'IVTPAH_NM', 'IVTPAM_NM', 'IVTPAMG_NM', 'NIVTPA_ID']]
    # ICD_ID = 1 (Infract)
    df_tpa = df_tpa[df_tpa.ICD_ID == 1]
    # IVTPATH_ID = 1 or 2 (1=This hospital, 2 = Another hospital)
    df_tpa['IVTPATH_ID'].loc[~df_tpa.IVTPATH_ID.isin(['1', '2'])] = np.nan
    # have validate date, hour, minutes ('IVTPA_DT', 'IVTPAH_NM', 'IVTPAM_NM')
    df_tpa['IVTPA_DT'] = pd.to_datetime(df_tpa['IVTPA_DT'], format='%Y-%M-%d', errors='ignore')
    df_tpa['IVTPAH_NM'].loc[(df_tpa.IVTPAH_NM < 0) & (df_tpa.IVTPAH_NM > 24)] = np.nan
    df_tpa['IVTPAM_NM'].loc[(df_tpa.IVTPAM_NM < 0) & (df_tpa.IVTPAM_NM > 60)] = np.nan
    # have IV-tPA mg
    df_tpa['IVTPAMG_NM'].loc[(df_tpa.IVTPAMG_NM < 0) & (df_tpa.IVTPAMG_NM > 101)] = np.nan
    # Not treat with IV t-PA (NIVTPA_ID) != 1, 2, 3
    df_tpa['NIVTPA_ID'].loc[df_tpa.NIVTPA_ID.isna()] = -1
    df_tpa['NIVTPA_ID'].loc[df_tpa.NIVTPA_ID > 0] = np.nan
    df_tpa.dropna(inplace=True)
    df_case['is_tpa'] = 0
    df_case['is_tpa'].loc[df_tpa.index] = 1
    if ~keep_cols:
        df_case.drop(['IVTPATH_ID', 'IVTPA_DT', 'IVTPAH_NM', 'IVTPAM_NM', 'IVTPAMG_NM', 'NIVTPA_ID'], inpace=True)
    print(df_case[df_case.is_tpa == 1].shape)
    return df_case


def nan_to_dont_know(df):
    # df_dgfa, df_fahi
    df = df.replace(np.nan, '2')
    return df

if __name__ == '__main__':
    # OFF_ID == 3 排除死亡及病危出院
