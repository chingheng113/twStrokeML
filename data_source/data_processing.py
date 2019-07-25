import pandas as pd
import numpy as np
import os
from data_source import selected_variables as sv
from data_source import mRS_validator as mv
from sklearn.preprocessing import Imputer
from functools import reduce

current_path = os.path.dirname(__file__)


def outliers_iqr(df, col):
    # http://colingorrie.github.io/outlier-detection.html
    quartile_1, quartile_3 = np.nanpercentile(df[col], [25, 75],)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)].index


def outlier_to_nan(df, columns):
    for col in columns:
        outlier_inx = outliers_iqr(df, col)
        df[col].loc[outlier_inx] = np.nan
    return df


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
        df_case.drop(['IVTPATH_ID', 'IVTPA_DT', 'IVTPAH_NM', 'IVTPAM_NM', 'IVTPAMG_NM', 'NIVTPA_ID'], axis=1, inplace=True)
    # print(df_case[df_case.is_tpa == 1].shape)
    return df_case


def day_in_hospital(df_case):
    in_date = pd.to_datetime(df_case['IH_DT'], format='%Y-%m-%d', errors='coerce')
    out_date = pd.to_datetime(df_case['OH_DT'], format='%Y-%m-%d', errors='coerce')
    day_diff = out_date - in_date
    df_case['in_hosptial_days'] = day_diff.dt.days
    df_case[df_case['in_hosptial_days'] < 0] = np.nan
    df_case.drop(['IH_DT', 'OH_DT'], axis=1, inplace=True)
    return df_case


def transfer_duration(df_case):
    df_case = df_case.fillna(value={'ONSETH_NM': 99, 'ONSETM_NM': 99, 'OTTIH_NM': 99, 'OTTIM_NM': 99})
    onset = df_case['ONSET_DT'].map(str)+'-'+df_case['ONSETH_NM'].astype(int).map(str)+'-'+df_case['ONSETM_NM'].astype(int).map(str)
    onset_day = pd.to_datetime(onset, format='%Y-%m-%d-%H-%M', errors='coerce')

    ot = df_case['OT_DT'].map(str) + '-' + df_case['OTTIH_NM'].astype(int).map(str) + '-' + df_case['OTTIM_NM'].astype(int).map(str)
    ot_day = pd.to_datetime(ot, format='%Y-%m-%d-%H-%M', errors='coerce')

    diff = ot_day - onset_day
    mins = diff.dt.seconds/60
    df_case['go_hospital_min'] = mins
    df_case[df_case['go_hospital_min'] < 0] = np.nan
    df_case.drop(['ONSETH_NM', 'ONSETM_NM', 'OTTIH_NM', 'OTTIM_NM', 'OT_DT'], axis=1, inplace=True)
    return df_case


def create_age(df):
    b_day = pd.to_datetime(df['YOB'], format='%Y', errors='coerce')
    onset_date = pd.to_datetime(df['ONSET_DT'], format='%Y-%m-%d', errors='coerce')
    age = np.floor((onset_date - b_day) / pd.Timedelta(days=365))
    df['onset_age'] = age
    df = df.drop(['YOB', 'ONSET_DT'], axis=1)
    return df


def nan_to_dont_know(df):
    # df_dgfa, df_fahi
    df.replace(to_replace={np.nan: 2}, inplace=True)
    return df


def nan_to_no_treatment(df_case):
    df_treat = df_case[sv.dcase_treat_bo].replace(to_replace={np.nan: 0})
    df_case[sv.dcase_treat_bo] = df_treat
    return df_case


def nan_to_no_compliant(df_case):
    df_treat = df_case[sv.dcase_complicaton_bo].replace(to_replace={np.nan: 0})
    df_case[sv.dcase_complicaton_bo] = df_treat
    return df_case


def nan_to_no_medication(df_case):
    df_treat = df_case[sv.dcase_med_bo].replace(to_replace={np.nan: 0})
    df_case[sv.dcase_med_bo] = df_treat
    return df_case


def complet_toast(df_case):
    df_case['TOAST_ID'].loc[df_case['ICD_ID'] != 1] = 0
    return df_case


def complete_cich_csah(df_case):
    df_case['CICH_ID'].loc[df_case['ICD_ID'] != 3] = 0
    df_case['CSAH_ID'].loc[df_case['ICD_ID'] != 4] = 0
    return df_case


def imputation_by_mean(df, cols):
    df[cols] = Imputer(missing_values=np.nan, strategy='mean', axis=0).fit_transform(df[cols])
    return df


def imputation_by_median(df, cols):
    df[cols] = Imputer(missing_values=np.nan, strategy='median', axis=0).fit_transform(df[cols])
    return df


if __name__ == '__main__':
    # CASEMCASE ==
    mcase = pd.read_csv(os.path.join('raw', 'CASEMCASE.csv'), na_values=np.nan)
    mcase = mcase[sv.mcase_id+sv.mcase_dt+sv.mcase_ca]
    mcase.replace(to_replace={'F': 0, 'M': 1}, inplace=True)

    # CASEDCASE ==
    dcase = pd.read_csv(os.path.join('raw', 'CASEDCASE.csv'), na_values=np.nan)
    dcase_selected_cols = sv.dcase_info_nm+sv.dcase_info_dt+sv.dcase_info_ca+sv.dcase_time_nm+sv.dcase_time_dt+\
                          sv.dcase_time_ca+sv.dcase_gcsv_nm+sv.dcase_icd_ca+sv.dcase_subtype_ca+\
                          sv.dcase_treat_bo+sv.dcase_med_bo+sv.dcase_complicaton_bo+sv.dcase_derterioation_bo+\
                          sv.dcase_lb_nm+sv.dcase_off_ca
    dcase = dcase[sv.ids+dcase_selected_cols]
    # OFF_ID == 3 排除死亡及病危出院
    dcase = dcase[dcase.OFF_ID == 3]
    dcase.drop(['OFF_ID'], axis=1, inplace=True)
    # 排除 other stroke
    dcase = dcase[dcase.ICD_ID != 99]
    # replace
    dcase.replace(to_replace={-999: np.nan, 'z': np.nan, 'Z': np.nan, 'N': 0, 'Y': 1}, inplace=True)
    # create is_tpa col
    dcase = is_tpa(dcase)
    # Days in hospital
    dcase = day_in_hospital(dcase)
    # duration of go-hospital
    dcase = transfer_duration(dcase)
    # Complete TOAST
    dcase = complet_toast(dcase)
    # Complet CICH & CSAH
    dcase = complete_cich_csah(dcase)
    # nan to no treatment
    dcase = nan_to_no_treatment(dcase)
    # non to no compliant
    dcase = nan_to_no_compliant(dcase)
    # non to no medication
    dcase = nan_to_no_medication(dcase)

    # Merge CASEMCASE and CASEDCASE
    df_final = pd.merge(dcase, mcase, on='ICASE_ID')
    df_final = create_age(df_final)

    # CASEDBMRS ==
    dbmrs = pd.read_csv(os.path.join('raw', 'CASEDBMRS.csv'), na_values=np.nan)
    dbmrs = dbmrs[sv.ids+sv.dbmrs_nm]
    dbmrs.replace(to_replace={-999: np.nan}, inplace=True)

    # CASEDCTMR ==
    dctmr = pd.read_csv(os.path.join('raw', 'CASEDCTMR.csv'), na_values=np.nan)
    dctmr = dctmr[sv.ids+sv.dctmr_nm]
    dctmr.replace(to_replace={'N': 0, 'Y': 1}, inplace=True)

    # CASEDGFA ==
    dgfa = pd.read_csv(os.path.join('raw', 'CASEDDGFA.csv'), na_values=np.nan)
    dgfa = dgfa[sv.ids+sv.ddgfa_ca]
    dgfa.replace(to_replace={-999: np.nan, 'z': np.nan, 'Z': np.nan}, inplace=True)
    dgfa = nan_to_dont_know(dgfa)

    # CASEDFAHI
    dfahi = pd.read_csv(os.path.join('raw', 'CASEDFAHI.csv'), na_values=np.nan)
    dfahi = dfahi[sv.ids+sv.dfahi_ca]
    dfahi.replace(to_replace={9: np.nan, 'Z': np.nan}, inplace=True)
    dfahi = nan_to_dont_know(dfahi)

    # CASEDNIHS
    dnihs = pd.read_csv(os.path.join('raw', 'CASEDNIHS.csv'), na_values=np.nan)
    dnihs = dnihs[sv.ids+sv.dnihs_nm]
    dnihs.replace(to_replace={-999: np.nan}, inplace=True)

    # CASEDRFUS
    drfur = pd.read_csv(os.path.join('raw', 'CASEDRFUR.csv'), na_values=np.nan)
    drfur = drfur[sv.ids+sv.drfur_ca+sv.drfur_bo+sv.drfur_nm]
    drfur['MRS_TX_1'].loc[~drfur.MRS_TX_1.isin([0, 1, 2, 3, 4, 5, 6])] = np.nan
    drfur['MRS_TX_3'].loc[~drfur.MRS_TX_3.isin([0, 1, 2, 3, 4, 5, 6])] = np.nan
    drfur.replace(to_replace={'N': 0, 'Y': 1}, inplace=True)

    # Merge together
    dfs = [df_final, dbmrs, dctmr, dgfa, dfahi, dnihs, drfur]
    df_final = reduce(lambda left, right: pd.merge(left, right, how='outer', on=sv.ids), dfs)
    df_final.to_csv('noImputation.csv', index=False)
    # imputation data
    imputation_cols = sv.dcase_lb_nm+sv.dcase_info_nm+['onset_age', 'SBP_NM', 'DBP_NM', 'BT_NM', 'HR_NM', 'RR_NM']
    df_final = outlier_to_nan(df_final, imputation_cols)
    df_final = imputation_by_mean(df_final, imputation_cols)

    # sorry~ ned to give up 'go_hospital_min' for more sample size...
    df_final.drop(['go_hospital_min'], inplace=True, axis=1)

    df_final.to_csv('TSR_2018_3m.csv', index=False)
    df_final.dropna(axis=0, inplace=True)
    df_final.to_csv('TSR_2018_3m_noMissing.csv', index=False)
    print(df_final.shape)
    # mRS validation
    df_final = mv.mRS_validate(df_final)
    df_final.to_csv('TSR_2018_3m_noMissing_validated.csv', index=False)
    print(df_final.shape)
    print('done')