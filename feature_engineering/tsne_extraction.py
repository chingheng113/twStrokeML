from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from my_utils import data_util, plot_fig, performance_util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


if __name__ == '__main__':
    seed = 7
    n_fold = 2
    n_class = 2
    if n_class == 2:
        id_data, x_data, y_data = data_util.get_poor_god('wholeset_Jim_nomissing_validated.csv')
        fn = 'reduced_dimension_2c'
    else:
        id_data, x_data, y_data = data_util.get_individual('wholeset_Jim_nomissing_validated.csv')
        fn = 'reduced_dimension_individual'
    kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    t_sne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0)
    for index, (train, test) in enumerate(kfold.split(x_data, y_data)):
        x_data_train = data_util.scale(x_data.iloc[train])
        t_sne.fit_transform(x_data_train)
        print('d')