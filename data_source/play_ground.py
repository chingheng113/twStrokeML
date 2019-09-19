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

    # CASEDCASE ==
    dcase = pd.read_csv(os.path.join('raw', 'processed_CASEDCASE.csv'), na_values=np.nan)
    # create is_tpa col
    dcase = is_tpa(dcase)
    case_col = ['ICASE_ID', 'IDCASE_ID', 'is_tpa', 'IVTPAMG_NM', 'TOAST_ID', 'THDA_FL', 'ONSET_DT',
                'ECGA_FL', 'THDH_FL', 'HB_NM', 'HCT_NM', 'PLATELET_NM', 'WBC_NM', 'PTINR_NM', 'PTT1_NM', 'ER_NM',
                'BUN_NM', 'CRE_NM', 'ALB_NM', 'CRP_NM', 'HBAC_NM', 'TCHO_NM', 'TG_NM', 'HDL_NM', 'LDL_NM', 'GOT_NM',
                'GPT_NM', 'SBP_NM', 'DBP_NM', 'HR_NM', 'FLOOKH_NM', 'FLOOKM_NM', 'IVTPAM_NM', 'DETHOH_FL', 'DETST_FL']
    dcase = dcase[case_col]
    # Merge CASEMCASE and CASEDCASE
    df_final = pd.merge(dcase, mcase, on='ICASE_ID')
    df_final = create_age(df_final)
    # CASEDRFUS
    drfur = pd.read_csv(os.path.join('raw', 'processed_CASEDRFUR.csv'), na_values=np.nan)
    drfur = drfur[['ICASE_ID', 'IDCASE_ID', 'MRS_TX_3']]
    # CASEDBMRS ==
    dbmrs = pd.read_csv(os.path.join('raw', 'processed_CASEDBMRS.csv'), na_values=np.nan)
    dbmrs = dbmrs[['ICASE_ID', 'IDCASE_ID', 'discharged_mrs']]
    # CASEDFAHI
    dfahi = pd.read_csv(os.path.join('raw', 'processed_CASEDFAHI.csv'), na_values=np.nan)
    # CASEDGFA ==
    dgfa = pd.read_csv(os.path.join('raw', 'processed_CASEDDGFA.csv'), na_values=np.nan)
    dgfa_cols = ['ICASE_ID', 'IDCASE_ID', 'PAD_ID', 'SM_ID', 'DM_ID', 'PTIA_ID', 'PCVACH_ID', 'PCVA_ID', 'HC_ID', 'HT_ID',
                 'PCVACI_ID', 'SMCP_ID', 'AL_ID', 'HD_ID']
    dgfa = dgfa[dgfa_cols]
    dfs = [df_final, dgfa, dfahi, drfur]
    df_final = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['ICASE_ID', 'IDCASE_ID']), dfs)
    df_final = df_final[df_final.is_tpa == 1]
    df_final.to_csv('xxxxxx.csv', index=False)
    print('done')