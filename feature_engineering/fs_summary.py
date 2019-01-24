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
        df['mean'] = df.drop(['f_index'], axis=1).mean(axis=1)
        # df.sort_values(by=['mean'], ascending=True, inplace=True)
        plot_features(df)
    print('Done')