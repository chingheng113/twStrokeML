import matplotlib
matplotlib.use('Agg')
from result import venn
import pandas as pd
import numpy as np
from my_utils import performance_util as pu
from my_utils import data_util
import os

df_all = data_util.load_all('TSR_2018_3m_noMissing_validated.csv')

mlp_err_h = pd.Series()
mlp_right_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp', 'fs', 'mlp_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_err_h = mlp_err_h.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    mlp_right_h = mlp_right_h.append(same_id)
mlp_err_h.drop_duplicates(inplace=True)
mlp_right_h.drop_duplicates(inplace=True)


mlp_cnn_err_h = pd.Series()
mlp_cnn_right_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp_cnn', 'fs', 'mlp_cnn_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_cnn_err_h = mlp_cnn_err_h.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    mlp_cnn_right_h = mlp_cnn_right_h.append(same_id)
mlp_cnn_err_h.drop_duplicates(inplace=True)
mlp_cnn_right_h.drop_duplicates(inplace=True)


svm_err_h = pd.Series()
svm_right_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('svm', 'fs', 'svm_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    svm_err_h = svm_err_h.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    svm_right_h = svm_right_h.append(same_id)
svm_err_h.drop_duplicates(inplace=True)
svm_right_h.drop_duplicates(inplace=True)


rf_err_h = pd.Series()
rf_right_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('rf', 'fs', 'rf_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    rf_err_h = rf_err_h.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    rf_right_h = rf_right_h.append(same_id)
rf_err_h.drop_duplicates(inplace=True)
rf_right_h.drop_duplicates(inplace=True)

all_wrong_h = set(mlp_cnn_err_h) & set(mlp_err_h) & set(svm_err_h) & set(rf_err_h)
wrong_icaseid_h = []
wrong_idcaseid_h = []
for v in list(all_wrong_h):
    ids = v.split('|')
    wrong_icaseid_h.append(ids[0])
    wrong_idcaseid_h.append(ids[1])
all_wrong_h_df = df_all.loc[(df_all['ICASE_ID'].isin(wrong_icaseid_h)) & (df_all['IDCASE_ID'].isin(wrong_idcaseid_h))]
print('a')
# labels = venn.get_labels([mlp_err_h, mlp_cnn_err_h, svm_err_h, rf_err_h], fill=['number'])
# fig, ax = venn.venn4(labels, names=['MLP', 'HANN', 'SVM', 'RF'])
# fig.savefig("hemo.png", dpi=300)


# ====
mlp_err_i = pd.Series()
mlp_right_i = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp', 'fs', 'mlp_fs_ischemic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_err_i = mlp_err_i.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    mlp_right_i = mlp_right_i.append(same_id)
mlp_err_i.drop_duplicates(inplace=True)
mlp_right_i.drop_duplicates(inplace=True)


mlp_cnn_err_i = pd.Series()
mlp_cnn_right_i = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp_cnn', 'fs', 'mlp_cnn_fs_ischemic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_cnn_err_i = mlp_cnn_err_i.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    mlp_cnn_right_i = mlp_cnn_right_i.append(same_id)
mlp_cnn_err_i.drop_duplicates(inplace=True)
mlp_cnn_right_i.drop_duplicates(inplace=True)


svm_err_i = pd.Series()
svm_right_i = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('svm', 'fs', 'svm_fs_ischemic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    svm_err_i = svm_err_i.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    svm_right_i = svm_right_i.append(same_id)
svm_err_i.drop_duplicates(inplace=True)
svm_right_i.drop_duplicates(inplace=True)


rf_err_i = pd.Series()
rf_right_i = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('rf', 'fs', 'rf_fs_ischemic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    rf_err_i = rf_err_i.append(diff_id)
    same = df.loc[df['predict_y'] == df['label']]
    same_id = same['ICASE_ID'].astype(str) + '|' + same['IDCASE_ID'].astype(str)
    rf_right_i = svm_right_i.append(same_id)
rf_err_i.drop_duplicates(inplace=True)
rf_right_i.drop_duplicates(inplace=True)

# labels = venn.get_labels([mlp_err_i, mlp_cnn_err_i, svm_err_i, rf_err_i], fill=['number'])
# fig, ax = venn.venn4(labels, names=['MLP', 'HANN', 'SVM', 'RF'])
# fig.savefig("isch.png", dpi=300)


print('done')