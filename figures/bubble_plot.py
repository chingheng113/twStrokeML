from my_utils import data_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bubble_plot(data, group_names):
    data_size = data.groupby(group_names).size()
    keys = data.groupby(group_names).groups.keys()
    x =[]
    y =[]
    for key in keys:
        x.append(key[0])
        y.append(key[1])
    plt.scatter(x, y, s=data_size, alpha=.5)
    plt.xlabel(group_names[0])
    plt.ylabel(group_names[1])


if __name__ == '__main__':
    b = 'bi_total'
    n = 'nihss_total'
    m = 'discharged_mrs'

    # -- Load Data
    df_3m = data_util.load_all('TSR_2018_3m_noMissing.csv')
    df_3m[b] = pd.DataFrame(np.sum(df_3m[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                          'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']], axis=1))
    df_3m[n] = pd.DataFrame(np.sum(df_3m[['NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out', 'NIHS_4_out',
                                          'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out',
                                          'NIHS_8_out',
                                          'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']], axis=1))

    # -- Load validated Data
    df_3m_validated = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')
    df_3m_validated['bi_total'] = pd.DataFrame(np.sum(df_3m_validated[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                                                       'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']], axis=1))
    df_3m_validated['nihss_total'] = pd.DataFrame(np.sum(df_3m_validated[['NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out', 'NIHS_4_out',
                                                                          'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out', 'NIHS_8_out',
                                                                          'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']], axis=1))

    # -- Plot
    # fig = plt.figure(figsize=(15, 5))
    bubble_plot(df_3m, [m, b])
    plt.ylabel('Discharge BI total score')
    plt.xlabel('Discharge mRS')
    plt.savefig('figure2-bubble', dpi=300)
    plt.show()