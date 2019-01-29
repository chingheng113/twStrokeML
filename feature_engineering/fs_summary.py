from my_utils import data_util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def cut_off(feature_list):
    for i in range(0, 10, 1):
        c_name = 'rf'+str(hold_out_round)
        feature_list_no_zero = feature_list.loc[feature_list[c_name] != 0]
        no_zero_importance = feature_list_no_zero[c_name].values
        cutoff = np.std(no_zero_importance)+np.min(no_zero_importance)
        list = feature_list.loc[feature_list[c_name] > cutoff]
        print(list.shape)
    return feature_list


if __name__ == '__main__':
    # Just get the feature names
    subtype = 'is'
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
    robust_f = cut_off(robust_f_df)




    # final_mean_importance = robust_f.drop(['f_index'], axis=1).mean(axis=1)
    # final_f = pd.DataFrame(data={'f_name': feature_names})
    # final_f['mean'] = final_mean_importance
    # #
    # selected_final_f = final_f.sort_values(by=['mean'], ascending=False).iloc[0:30,:]
    # selected_final_f.to_csv('selected_features_'+sub_class+'.csv', sep=',', encoding='utf-8', index=False)
    # plot_features(final_f)
    print('Done')
