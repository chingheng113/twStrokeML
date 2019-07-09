import matplotlib
matplotlib.use('Agg')
from result import venn
import pandas as pd
import numpy as np
from my_utils import performance_util as pu
import os


mlp_err_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp', 'fs', 'mlp_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_err_h = mlp_err_h.append(diff_id)
mlp_err_h.drop_duplicates(inplace=True)


mlp_cnn_err_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('mlp_cnn', 'fs', 'mlp_cnn_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    mlp_cnn_err_h = mlp_cnn_err_h.append(diff_id)
mlp_cnn_err_h.drop_duplicates(inplace=True)


svm_err_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('svm', 'fs', 'svm_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    svm_err_h = svm_err_h.append(diff_id)
svm_err_h.drop_duplicates(inplace=True)


rf_err_h = pd.Series()
for i in range(10):
    df = pd.read_csv(os.path.join('rf', 'fs', 'rf_fs_hemorrhagic_h_'+str(i)+'_hold.csv'))
    df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
    diff = df.loc[df['predict_y'] != df['label']]
    diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
    rf_err_h = rf_err_h.append(diff_id)
rf_err_h.drop_duplicates(inplace=True)

x = set(mlp_cnn_err_h) & set(mlp_err_h) & set(svm_err_h) & set(rf_err_h)
print(len(x))
for v in list(x):
    print(v.split('|'))

# labels = venn.get_labels([mlp_err_h, mlp_cnn_err_h, svm_err_h, rf_err_h], fill=['number'])
# fig, ax = venn.venn4(labels, names=['MLP', 'HANN', 'SVM', 'RF'])
# fig.savefig("hemo.png", dpi=300)


# ====
# mlp_err_i = pd.Series()
# for i in range(10):
#     df = pd.read_csv(os.path.join('mlp', 'fs', 'mlp_fs_ischemic_h_'+str(i)+'_hold.csv'))
#     df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
#     diff = df.loc[df['predict_y'] != df['label']]
#     diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
#     mlp_err_i = mlp_err_i.append(diff_id)
# mlp_err_i.drop_duplicates(inplace=True)
#
#
# mlp_cnn_err_i = pd.Series()
# for i in range(10):
#     df = pd.read_csv(os.path.join('mlp_cnn', 'fs', 'mlp_cnn_fs_ischemic_h_'+str(i)+'_hold.csv'))
#     df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
#     diff = df.loc[df['predict_y'] != df['label']]
#     diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
#     mlp_cnn_err_i = mlp_cnn_err_i.append(diff_id)
# mlp_cnn_err_i.drop_duplicates(inplace=True)
#
#
# svm_err_i = pd.Series()
# for i in range(10):
#     df = pd.read_csv(os.path.join('svm', 'fs', 'svm_fs_ischemic_h_'+str(i)+'_hold.csv'))
#     df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
#     diff = df.loc[df['predict_y'] != df['label']]
#     diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
#     svm_err_i = svm_err_i.append(diff_id)
# svm_err_i.drop_duplicates(inplace=True)
#
#
# rf_err_i = pd.Series()
# for i in range(10):
#     df = pd.read_csv(os.path.join('rf', 'fs', 'rf_fs_ischemic_h_'+str(i)+'_hold.csv'))
#     df['predict_y'] = pu.labelize(df[['0', '1']].values).astype(int)
#     diff = df.loc[df['predict_y'] != df['label']]
#     diff_id = diff['ICASE_ID'].astype(str)+'|'+diff['IDCASE_ID'].astype(str)
#     rf_err_i = rf_err_i.append(diff_id)
# rf_err_i.drop_duplicates(inplace=True)
#
#
# labels = venn.get_labels([mlp_err_i, mlp_cnn_err_i, svm_err_i, rf_err_i], fill=['number'])
# fig, ax = venn.venn4(labels, names=['MLP', 'HANN', 'SVM', 'RF'])
# fig.savefig("isch.png", dpi=300)


print('done')