import pandas as pd
import numpy as np
from functools import reduce
import os
current_path = os.path.dirname(__file__)

def create_age(df):
    b_day = pd.to_datetime(df['YOB'], format='%Y', errors='coerce')
    onset_date = pd.to_datetime(df['ONSET_DT'], format='%Y/%m/%d', errors='coerce')
    age = np.floor((onset_date - b_day) / pd.Timedelta(days=365))
    df['onset_age'] = age
    df = df.drop(['YOB', 'ONSET_DT'], axis=1)
    return df

def create_age2(df):
    b_day = pd.to_datetime(df['YOB'], format='%Y-%m-%d', errors='coerce')
    onset_date = pd.to_datetime(df['ONSET_DT'], format='%m/%d/%Y', errors='coerce')
    age = np.floor((onset_date - b_day) / pd.Timedelta(days=365))
    df['onset_age'] = age
    df = df.drop(['YOB', 'ONSET_DT'], axis=1)
    return df

def is_tpa(df_case):
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

    return df_case

if __name__ == '__main__':
    # print([x for x in selected_variables1 if x not in case_col])
    # CASEDCASE ==
    mcase = pd.read_csv(os.path.join('raw', 'processed_CASEMCASE.csv'), na_values=np.nan)
    mcase2 = pd.read_csv(os.path.join('raw', 'CASEMCASE_2019.csv'), na_values=np.nan)

    # CASEDCASE ==
    dcase = pd.read_csv(os.path.join('raw', 'processed_CASEDCASE.csv'), na_values=np.nan)
    dcase2 = pd.read_csv(os.path.join('raw', 'CASEDCASE_2019.csv'), na_values=np.nan)
    # Merge CASEMCASE and CASEDCASE
    df_final = pd.merge(dcase, mcase, on='ICASE_ID')
    df_final = create_age(df_final)

    df_final2 = pd.merge(dcase2, mcase2, on='ICASE_ID')
    df_final2 = create_age2(df_final2)
    df_final2 = df_final2[(df_final2.onset_age < 100) & (df_final2.onset_age > 18)]
    df_final2 = df_final2[df_final2.ICD_ID.isin([1.0, 2.0, 3.0, 4.0])]

    df_diff = pd.merge(df_final, df_final2, on=['ICASE_ID',	'IDCASE_ID'], how='right', indicator=True)

    df_final.to_csv('2018.csv')
    df_final2.to_csv('2019.csv')
    df_diff.to_csv('2000.csv')

    print('done')