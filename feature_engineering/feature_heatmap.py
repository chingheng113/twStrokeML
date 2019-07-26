import seaborn as sns
import pandas as pd
import numpy as np
from my_utils import data_util
import matplotlib.pyplot as plt
from collections import Counter


def get_final_features(feature_df, feature_names):
    # 10 hold-out rounds
    all_selected_features = []
    for i in range(0, 10, 1):
        c_name = 'rf'+str(i)
        feature_list = feature_df[['f_index', c_name]]
        feature_list_no_zero = feature_list.loc[feature_list[c_name] != 0]
        no_zero_importance = feature_list_no_zero[c_name].values
        cutoff = np.std(no_zero_importance)+np.min(no_zero_importance)
        selected_f = feature_list.loc[feature_list[c_name] > cutoff]
        selected_f.sort_values(by=[c_name], inplace=True, ascending=False)
        selected_f_names = feature_names[selected_f['f_index'].values]
        if i == 0:
            all_selected_features = selected_f_names
        else:
            all_selected_features = np.append(all_selected_features, selected_f_names)
    feature_dict = Counter(all_selected_features)
    temp = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['value']).reset_index()
    final_features = pd.DataFrame(data=temp['value'].values, index=temp['index'].values, columns=['count'])
    final_features['name'] = final_features.index
    sorted_final_features = final_features.sort_values(by=['count', 'name'], ascending=False)
    sorted_final_features.drop(['name'], axis=1, inplace=True)
    return sorted_final_features


def get_robust_features(subtype):
    for hold_out_round in range(0, 10, 1):
        if subtype == 'is':
            sub_class = 'ischemic'
        else:
            sub_class = 'hemorrhagic'
        df = pd.read_csv('f_'+subtype+'_'+str(hold_out_round)+'.csv', encoding='utf8')
        mean_importance = df.drop(['f_index'], axis=1).mean(axis=1)
        if hold_out_round == 0:
            robust_f_df = pd.DataFrame(data={'f_index': df['f_index']})
            robust_f_df['rf'+str(hold_out_round)] = mean_importance
        else:
            robust_f_df['rf'+str(hold_out_round)] = mean_importance
    return robust_f_df


def get_feature_names():
    id_train_all, x_train_all, y_train_all = data_util.get_poor_god(
        'training_is_0.csv', sub_class='ischemic')
    x_train_all.rename(columns={'MRS_TX_1': '30-day mRS', 'discharged_mrs': 'Discharge mRS', 'Toilet_use': 'Toilet use',
                                'Bowel_control': 'Bowel control', 'Bladder_control': 'Bladder control',
                                'TRMNG_FL': 'Nasogastric tube', 'TRMRE_FL': 'Rehab', 'OFFDT_ID_1': 'Discharge to Home',
                                'NIHS_6aL_out': 'Discharge NIHSS 6aL', 'NIHS_6aL_in': 'Admission NIHSS 6aL',
                                'NIHS_6bR_out': 'Discharge NIHSS 6bR', 'NIHS_10_out': 'Discharge NIHSS 10',
                                'NIHS_5aL_out': 'Discharge NIHSS 5aL', 'NIHS_5bR_out': 'Discharge NIHSS 5bR',
                                'NIHS_1b_out': 'Discharge NIHSS 1b', 'NIHS_9_out': 'Discharge NIHSS 9',
                                'NIHS_5aL_in': 'Admission NIHSS 5aL', 'NIHS_1b_in': 'Admission NIHSS 1b'}
                       , inplace=True)
    return x_train_all.columns.values


def plot_heatmap(is_df, he_df):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
    sns.set(font_scale=1.1)
    plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=1, hspace=None)
    plt.subplot(121)
    plt.title('Ischemic stroke', fontsize=14)
    sns.heatmap(is_df, annot=True, fmt="d", cmap='viridis', linewidths=0.3)
    plt.subplot(122)
    plt.title('Hemorrhagic stroke', fontsize=14)
    sns.heatmap(he_df, annot=True, fmt="g", cmap='viridis', linewidths=0.3)
    plt.show()


if __name__ == '__main__':
    feature_names = get_feature_names()
    is_robust_f_df = get_robust_features('is')
    is_final_features = get_final_features(is_robust_f_df, feature_names)
    he_robust_f_df = get_robust_features('he')
    he_final_features = get_final_features(he_robust_f_df, feature_names)
    plot_heatmap(is_final_features, he_final_features)
    print('done')
