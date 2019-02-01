from my_utils import data_util
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter



def plot_features(df):
    importances = df['mean'].values
    indices = np.argsort(df['mean'])
    # Top 30
    indices = indices[-30:]
    plt.figure(1)
    plt.barh(range(len(indices)), df['mean'].loc[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices], fontsize=6)
    plt.xlabel('Average Importance')
    plt.show()


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
    # use to draw heatmap
    return list(feature_dict.keys())


if __name__ == '__main__':
    # Just get the feature names
    subtype = 'he'
    if subtype == 'is':
        sub_class = 'ischemic'
    else:
        sub_class = 'hemorrhagic'
    id_train_all, x_train_all, y_train_all = data_util.get_poor_god(
        'training_' + subtype + '_0.csv', sub_class=sub_class)
    feature_names = x_train_all.columns.values
    #
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

    final_features = get_final_features(robust_f_df, feature_names)
    final_features = np.concatenate([['f_name'], final_features])
    np.savetxt('selected_features' + '_' + sub_class + '.csv', final_features, delimiter=',', fmt='%s')

    print('Done')
