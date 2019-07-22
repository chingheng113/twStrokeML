import numpy as np
import pandas as pd


def mRS_validate(df):
    df['bi_total'] = pd.DataFrame(np.sum(df[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                             'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']], axis=1))
    df['nihss_total'] = pd.DataFrame(np.sum(df[['NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out', 'NIHS_4_out',
                                            'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out', 'NIHS_8_out',
                                            'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']], axis=1))
    df['nihss_total_in'] = pd.DataFrame(np.sum(df[['NIHS_1a_in', 'NIHS_1b_in', 'NIHS_1c_in', 'NIHS_2_in', 'NIHS_3_in', 'NIHS_4_in',
                                                   'NIHS_5aL_in', 'NIHS_5bR_in', 'NIHS_6aL_in', 'NIHS_6bR_in', 'NIHS_7_in', 'NIHS_8_in',
                                                   'NIHS_9_in', 'NIHS_10_in', 'NIHS_11_in']], axis=1))
    # Doing clinical logic validation
    df_valied = logic_validate(df)
    # Doing LOWESS regression validation
    df_valied = lowess_validate_on_BI(df_valied)

    df_valied = df_valied.drop(['bi_total', 'nihss_total', 'nihss_total_in'], axis=1)
    return df_valied

def logic_validate(df):
    # print(df.shape)
    df = df[~((df['Mobility'] == 0) & (df['Stairs'] != 0))]
    # print(df.shape)
    df = df[~((df['Stairs'] == 10) & (df['NIHS_6aL_out'] == 4) & (df['NIHS_6bR_out'] == 4))]
    # print(df.shape)
    df = df[~((df['discharged_mrs'] != 5) & (df['NIHS_1a_out'] == 3))]
    # print(df.shape)
    df = df[~(df['nihss_total'] > 39)]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1b_out'] != 2))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1c_out'] != 2))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_4_out'] != 3))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5aL_out'] != 4))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5bR_out'] != 4))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6aL_out'] != 4))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6bR_out'] != 4))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_7_out'] != 0))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_8_out'] != 2))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_9_out'] != 3))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_10_out'] != 2))]
    # print(df.shape)
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_11_out'] != 0))]
    # print(df.shape)
    df = df[~((df['bi_total'] == 0) & (df['discharged_mrs'] < 5))]
    # print(df.shape)
    return df


def lowess_validate_on_BI(df):
    '''the boundaries are given by lowess_clean.R'''
    upper_bound = [114.78681, 113.96771, 109.56282,  87.51552,  55.93302,  18.62162]
    lower_bound = [84.64237,  83.82327,  79.41838,  57.37109,  25.78858, -11.52281]
    # for i in range(6):
    #     df = df[~((df['discharged_mrs'] == i) & (df['bi_total'] > lower_bound[i]) & (df['bi_total'] < upper_bound[i]))]
    df0 = df[(df['discharged_mrs'] == 0) & (df['bi_total'] > lower_bound[0]) & (df['bi_total'] < upper_bound[0])]
    df1 = df[(df['discharged_mrs'] == 1) & (df['bi_total'] > lower_bound[1]) & (df['bi_total'] < upper_bound[1])]
    df2 = df[(df['discharged_mrs'] == 2) & (df['bi_total'] > lower_bound[2]) & (df['bi_total'] < upper_bound[2])]
    df3 = df[(df['discharged_mrs'] == 3) & (df['bi_total'] > lower_bound[3]) & (df['bi_total'] < upper_bound[3])]
    df4 = df[(df['discharged_mrs'] == 4) & (df['bi_total'] > lower_bound[4]) & (df['bi_total'] < upper_bound[4])]
    df5 = df[(df['discharged_mrs'] == 5) & (df['bi_total'] > lower_bound[5]) & (df['bi_total'] < upper_bound[5])]
    df_final = pd.concat([df0, df1, df2, df3, df4, df5])
    return df_final