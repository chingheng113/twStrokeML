from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from my_utils import data_util, plot_fig, performance_util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


def plot_all_features(importances, feature_names):
    indices = np.argsort(importances)
    for feature in zip(feature_names, rf.feature_importances_):
        print(feature)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel('Relative Importance')
    plt.show()


if __name__ == '__main__':
    # https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    # wholeset_Jim_nomissing_validated.csv
    seed = 7
    n_fold = 2
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
    feature_names = x_data.columns.values

    # Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than XX
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=seed)
    sfm = SelectFromModel(rf, threshold=5e-3)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        rf.fit(data_util.scale(x_data.iloc[train]), y_data.iloc[train])
        importances = rf.feature_importances_
        # plot_all_features(importances, feature_names)
        # Train the selector
        sfm.fit(x_data.iloc[train], y_data.iloc[train])
        # Print the names of the most important features
        for feature_list_index in sfm.get_support(indices=True):
            print(feature_names[feature_list_index])
        # Transform the data to create a new dataset containing only the most important features
        # X_important_train = x_data.iloc[train][x_data.iloc[train].columns[sfm.get_support(indices=True)]]
        # X_important_train = sfm.transform(x_data.iloc[train])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    id_data_all, x_data_all, y_data_all = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
    x_data_selected = x_data_all[x_data_all.columns[sfm.get_support(indices=True)]]
    data_fs = pd.concat([id_data_all, x_data_selected, y_data_all], axis=1)
    data_util.save_dataframe_to_csv(data_fs, 'wholeset_Jim_nomissing_validated_fs')
    print(data_fs.shape)



