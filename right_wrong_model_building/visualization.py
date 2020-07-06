import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data1 = pd.read_csv(os.path.join('..', 'result', 'all_right_wrong_i.csv'))
data2 = pd.read_csv(os.path.join('..', 'data_source', 'TSR_2018_3m_noMissing_validated.csv'))
data = pd.merge(data1[['ICASE_ID', 'IDCASE_ID', 'ctype']], data2, how='inner', on=['ICASE_ID', 'IDCASE_ID'])

right_data = data[data.ctype == 1]
wrong_data = data[data.ctype == 0]

right_data = right_data.assign(BI_total=np.sum(right_data[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming',
                                                           'Mobility', 'Stairs', 'Dressing', 'Bowel_control',
                                                           'Bladder_control']], axis=1))

right_data = right_data.assign(NIHS_out_subtotal=np.sum(right_data[['NIHS_5aL_out', 'NIHS_6aL_out', 'NIHS_5bR_out']], axis=1))

wrong_data = wrong_data.assign(BI_total=np.sum(wrong_data[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming',
                                                           'Mobility', 'Stairs', 'Dressing', 'Bowel_control',
                                                           'Bladder_control']], axis=1))

wrong_data = wrong_data.assign(NIHS_out_subtotal=np.sum(wrong_data[['NIHS_5aL_out', 'NIHS_6aL_out', 'NIHS_5bR_out']], axis=1))

# --
print(np.mean(right_data[right_data.MRS_TX_1 == 3].BI_total), np.mean(right_data[right_data.MRS_TX_1 == 3].NIHS_out_subtotal))
print(np.mean(wrong_data[wrong_data.MRS_TX_1 == 3].BI_total), np.mean(wrong_data[wrong_data.MRS_TX_1 == 3].NIHS_out_subtotal))
# plt.boxplot([right_data[right_data.MRS_TX_1 == 3].BI_total, wrong_data[wrong_data.MRS_TX_1 == 3].BI_total])
# plt.show()
# plt.boxplot([right_data[right_data.MRS_TX_1 == 3].NIHS_out_subtotal, wrong_data[wrong_data.MRS_TX_1 == 3].NIHS_out_subtotal])
# plt.show()

plt.boxplot([right_data.in_hosptial_days, wrong_data.in_hosptial_days])
plt.show()
plt.boxplot([right_data.ER_NM, wrong_data.ER_NM])
plt.show()
print(np.mean(right_data.in_hosptial_days), np.mean(wrong_data.in_hosptial_days))
print(np.mean(right_data.ER_NM), np.mean(wrong_data.ER_NM))
# --
right_5al_count = right_data.groupby(['NIHS_5aL_out']).size()
right_5al_percent = right_5al_count / np.sum(right_5al_count)

wrong_5al_count = wrong_data.groupby(['NIHS_5aL_out']).size()
wrong_5al_percent = wrong_5al_count / np.sum(wrong_5al_count)

bar_df = pd.DataFrame(data={'Right': right_5al_percent, 'Wrong': wrong_5al_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('NIHS_5aL_out')
plt.show()

# --
right_mrs_count = right_data.groupby(['discharged_mrs']).size()
right_mrs_percent = right_mrs_count / np.sum(right_mrs_count)

wrong_mrs_count = wrong_data.groupby(['discharged_mrs']).size()
wrong_mrs_percent = wrong_mrs_count / np.sum(wrong_mrs_count)

bar_df = pd.DataFrame(data={'Right': right_mrs_percent, 'Wrong': wrong_mrs_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('discharged_mrs')
plt.show()
# --
right_tra_count = right_data.groupby(['Grooming']).size()
right_tra_percent = right_tra_count / np.sum(right_tra_count)

wrong_tra_count = wrong_data.groupby(['Grooming']).size()
wrong_tra_percent = wrong_tra_count / np.sum(wrong_tra_count)

bar_df = pd.DataFrame(data={'Right': right_tra_percent, 'Wrong': wrong_tra_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('Grooming')
plt.show()
# --
right_mob_count = right_data.groupby(['Stairs']).size()
right_mob_percent = right_mob_count / np.sum(right_mob_count)

wrong_mob_count = wrong_data.groupby(['Stairs']).size()
wrong_mob_percent = wrong_mob_count / np.sum(wrong_mob_count)

bar_df = pd.DataFrame(data={'Right': right_mob_percent, 'Wrong': wrong_mob_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('Stairs')
plt.show()
# --
right_fed_count = right_data.groupby(['Feeding']).size()
right_fed_percent = right_fed_count / np.sum(right_fed_count)

wrong_fed_count = wrong_data.groupby(['Feeding']).size()
wrong_fed_percent = wrong_fed_count / np.sum(wrong_fed_count)

bar_df = pd.DataFrame(data={'Right': right_fed_percent, 'Wrong': wrong_fed_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('feeding')
plt.show()
# --
right_bat_count = right_data.groupby(['Bathing']).size()
right_bat_percent = right_bat_count / np.sum(right_bat_count)

wrong_bat_count = wrong_data.groupby(['Bathing']).size()
wrong_bat_percent = wrong_bat_count / np.sum(wrong_bat_count)

bar_df = pd.DataFrame(data={'Right': right_bat_percent, 'Wrong': wrong_bat_percent})
fig = plt.figure()
bar_df.plot.bar()
plt.title('bathing')
plt.show()



print('done')


# ('MRS_TX_1 <= 1.00', 0.3287786847821817)
# ('10.00 < Transfers <= 15.00', 0.09230418167930607)
# ('5.00 < Feeding <= 10.00', -0.0873625093986046)
# ('10.00 < Mobility <= 15.00', -0.08277820240015978)
# ('0.00 < Bathing <= 5.00', -0.06277646051968833)