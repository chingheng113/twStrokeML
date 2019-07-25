# https://scikit-learn.org/stable/modules/feature_selection.html
# https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from my_utils import data_util

if __name__ == '__main__':
    subtype = 'he'
    # hold_out_round = 1
    for hold_out_round in range(0, 10, 1):
        if subtype == 'is':
            sub_class = 'ischemic'
        else:
            sub_class = 'hemorrhagic'
        id_train_all, x_train_all, y_train_all = data_util.get_poor_god('training_' + subtype + '_' + str(hold_out_round) + '.csv', sub_class=sub_class)
        feature_names = x_train_all.columns.values
        forest = ExtraTreesClassifier()
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=hold_out_round)
        for index, (train, test) in enumerate(kfold.split(x_train_all, y_train_all)):
            x_train = data_util.scale(x_train_all.iloc[train])
            y_train = y_train_all.iloc[train]
            forest.fit(x_train, y_train)
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]
            # Print the feature ranking
            imp = []
            print("Feature ranking:")
            for i in range(x_train.shape[1]):
                print("%d. feature %d (%f) %s" % (i + 1, indices[i], importances[indices[i]], feature_names[indices[i]]))
                imp.append(importances[indices[i]])
            importance_df = pd.DataFrame(data={'f_index': indices, 'score': imp})
            importance_df.sort_values(by=['f_index'], inplace=True)
            importance_df.reset_index(drop=True, inplace=True)
            if index == 0:
                result_df = importance_df.rename(columns={'score': 'score_0'})
            else:
                result_df['score_'+str(index)] = importance_df['score']
        result_df.to_csv('f_'+subtype+'_'+str(hold_out_round)+'.csv', sep=',', encoding='utf-8', index=False)
        print('Done')
