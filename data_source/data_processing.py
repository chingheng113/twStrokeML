import pandas as pd
import numpy as np
import os
from data_source import selected_variables as sv
current_path = os.path.dirname(__file__)


def outliers_iqr(ys):
    # http://colingorrie.github.io/outlier-detection.html
    quartile_1, quartile_3 = np.nanpercentile(ys, [25, 75],)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


def outlier_to_nan(df, columns):
    for col in columns:
        outlier_inx = outliers_iqr(df[col])
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
    print(df_case[df_case.is_tpa == 1].shape)
    return df_case


def nan_to_dont_know(df):
    # df_dgfa, df_fahi
    df = df.replace(np.nan, '2')
    return df


if __name__ == '__main__':
    dcase = pd.read_csv(os.path.join('raw', 'CASEDCASE.csv'))
    dcase_selected_cols = sv.dcase_info_nm+sv.dcase_info_dt+sv.dcase_info_ca+sv.dcase_time_nm+sv.dcase_time_dt+\
                          sv.dcase_time_ca+sv.dcase_gcsv_nm+sv.dcase_icd_ca+sv.dcase_subtype_ca+sv.dcase_heart_bo+\
                          sv.dcase_treat_bo+sv.dcase_med_bo+sv.dcase_complicaton_bo+sv.dcase_derterioation_bo+\
                          sv.dcase_ecg_ca+sv.dcase_lb_nm+sv.dcase_off_ca
    dcase = dcase[sv.ids+dcase_selected_cols]
    # OFF_ID == 3 排除死亡及病危出院
    dcase = dcase[dcase.OFF_ID == 3]
    dcase.replace(to_replace={-999: np.nan, 'z': np.nan, 'N': 0, 'Y': 1}, inplace=True)
    # create is_tpa col
    dcase = is_tpa(dcase)
    # IQR
    dcase = outlier_to_nan(dcase, sv.dcase_lb_nm)


    

     # age
    print('done')