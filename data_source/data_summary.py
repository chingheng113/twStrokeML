import pandas as pd
import numpy as np
from my_utils import data_util

if __name__ == '__main__':
    df = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    df = df[df['ICD_ID_99.0'] != 1]
    df['bi_total'] = pd.DataFrame(np.sum(df[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                             'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']], axis=1))
    df['nihss_total'] = pd.DataFrame(np.sum(df[['NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out', 'NIHS_4_out',
                                                'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out', 'NIHS_8_out',
                                                'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']], axis=1))

    total_n = df.shape[0]
    female_df = df[df['GENDER_TX'] == 0]
    male_df = df[df['GENDER_TX'] == 1]

    # female_n = female_df.shape[0]
    # female_p = round(female_n/total_n, 2)
    # male_n = male_df.shape[0]
    # male_p = round(male_n/total_n, 2)
    #
    # female_age_mean = round(np.mean(female_df['onset_age']), 1)
    # female_age_std = round(np.std(female_df['onset_age']), 1)
    # male_age_mean = round(np.mean(male_df['onset_age']), 1)
    # male_age_std = round(np.std(male_df['onset_age']), 1)

    # female_good = female_df[female_df['discharged_mrs'] < 3].shape[0]
    # female_good_p = round(female_good/total_n, 2)
    # female_poor = female_df[female_df['discharged_mrs'] > 2].shape[0]
    # female_poor_p = round(female_poor/total_n, 2)
    #
    # male_good = male_df[male_df['discharged_mrs'] < 3].shape[0]
    # male_good_p = round(male_good/total_n, 2)
    # male_poor = male_df[male_df['discharged_mrs'] > 2].shape[0]
    # male_poor_p = round(male_poor/total_n, 2)
    #
    # f_bi_i = female_df [(79 < female_df['bi_total']) & (female_df['bi_total'] < 101)].shape[0]
    # f_bi_i_p = round(f_bi_i/total_n, 2)
    # f_bi_m = female_df [(59 < female_df['bi_total']) & (female_df['bi_total'] < 80)].shape[0]
    # f_bi_m_p = round(f_bi_m/total_n, 2)
    # f_bi_p = female_df [(39 < female_df['bi_total']) & (female_df['bi_total'] < 60)].shape[0]
    # f_bi_p_p = round(f_bi_p/total_n, 2)
    # f_bi_v = female_df [(19 < female_df['bi_total']) & (female_df['bi_total'] < 40)].shape[0]
    # f_bi_v_p = round(f_bi_v/total_n, 2)
    # f_bi_t = female_df [female_df['bi_total'] < 20].shape[0]
    # f_bi_t_p = round(f_bi_p/total_n,2)
    #
    # m_bi_i = male_df [(79 < male_df['bi_total']) & (male_df['bi_total'] < 101)].shape[0]
    # m_bi_i_p = round(m_bi_i/total_n, 2)
    # m_bi_m = male_df [(59 < male_df['bi_total']) & (male_df['bi_total'] < 80)].shape[0]
    # m_bi_m_p = round(m_bi_m/total_n, 2)
    # m_bi_p = male_df [(39 < male_df['bi_total']) & (male_df['bi_total'] < 60)].shape[0]
    # m_bi_p_p = round(m_bi_p/total_n, 2)
    # m_bi_v = male_df [(19 < male_df['bi_total']) & (male_df['bi_total'] < 40)].shape[0]
    # m_bi_v_p = round(m_bi_v/total_n, 2)
    # m_bi_t = male_df [male_df['bi_total'] < 20].shape[0]
    # m_bi_t_p = round(m_bi_p/total_n,2)

    f_ni_0 = female_df[female_df['nihss_total'] == 0].shape[0]
    f_ni_0_p = round(f_ni_0/total_n, 2)
    f_ni_1 = female_df[(0 < female_df['nihss_total']) & (female_df['nihss_total'] < 5)].shape[0]
    f_ni_1_p = round(f_ni_1/total_n, 2)
    f_ni_2 = female_df[(4 < female_df['nihss_total']) & (female_df['nihss_total'] < 16)].shape[0]
    f_ni_2_p = round(f_ni_2/total_n, 2)
    f_ni_3 = female_df[(15 < female_df['nihss_total']) & (female_df['nihss_total'] < 21)].shape[0]
    f_ni_3_p = round(f_ni_3/total_n, 2)
    f_ni_4 = female_df[20 < female_df['nihss_total']].shape[0]
    f_ni_4_p = round(f_ni_4/total_n, 2)

    m_ni_0 = male_df[male_df['nihss_total'] == 0].shape[0]
    m_ni_0_p = round(m_ni_0/total_n, 2)
    m_ni_1 = male_df[(0 < male_df['nihss_total']) & (male_df['nihss_total'] < 5)].shape[0]
    m_ni_1_p = round(m_ni_1/total_n, 2)
    m_ni_2 = male_df[(4 < male_df['nihss_total']) & (male_df['nihss_total'] < 16)].shape[0]
    m_ni_2_p = round(m_ni_2/total_n, 2)
    m_ni_3 = male_df[(15 < male_df['nihss_total']) & (male_df['nihss_total'] < 21)].shape[0]
    m_ni_3_p = round(m_ni_3/total_n, 2)
    m_ni_4 = male_df[20 < male_df['nihss_total']].shape[0]
    m_ni_4_p = round(m_ni_4/total_n, 2)
    print('done')
